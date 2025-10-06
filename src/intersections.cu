#include "defines.h"
#include "intersections.h"
#include "utilities.h"

constexpr bool __device__ ChannelCheck[5][5] = {
    {true, true, true, true, true},
    {true, false, false, false, false},
    {true, false, false, false, false},
    {true, true, true, true, false},
    {false, false, false, false, false},
};

__host__ __device__ float AABBIntersect(glm::vec3 minCorner, glm::vec3 maxCorner, const Ray& r)
{
    glm::vec3 invDir = glm::vec3(1.0) / r.direction;

    glm::vec3 f = (maxCorner - r.origin) * invDir;
    glm::vec3 n = (minCorner - r.origin) * invDir;

    glm::vec3 tmax = max(f, n);
    glm::vec3 tmin = min(f, n);

    float t1 = glm::min(tmax.x, glm::min(tmax.y, tmax.z));
    float t0 = glm::max(tmin.x, glm::max(tmin.y, tmin.z));

    return (t1 >= t0) ? (t0 > 0.f ? t0 : t1) : -1.0;
}

__host__ __device__ float SphereIntersect(float rad, glm::vec3 pos, const Ray& r)
{
    glm::vec3 op = pos - r.origin;
    float b = glm::dot(op, r.direction);
    float det = b * b - glm::dot(op, op) + rad * rad;
    if (det < 0.0)
        return INFINITY;

    det = sqrt(det);
    float t1 = b - det;
    if (t1 > 0.001)
        return t1;

    float t2 = b + det;
    if (t2 > 0.001)
        return t2;

    return INFINITY;
}

__host__ __device__ float RectIntersect(glm::vec3 pos, glm::vec3 u, glm::vec3 v, glm::vec4 plane, const Ray& r)
{
    glm::vec3 n = glm::vec3(plane);
    float dt = glm::dot(r.direction, n);
    float t = (plane.w - glm::dot(n, r.origin)) / dt;

    if (t > EPSILON) {
        glm::vec3 p = r.origin + r.direction * t;
        glm::vec3 vi = p - pos;
        float a1 = glm::dot(u, vi);
        if (a1 >= 0.0 && a1 <= 1.0) {
            float a2 = glm::dot(v, vi);
            if (a2 >= 0.0 && a2 <= 1.0)
                return t;
        }
    }
    return INFINITY;
}

__host__ __device__ float TriangleIntersect(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, const Ray& r, glm::vec3& bary)
{
    glm::vec3 e0 = v1 - v0;
    glm::vec3 e1 = v2 - v0;
    glm::vec3 pv = glm::cross(r.direction, e1);
    float det = glm::dot(e0, pv);

    if (glm::abs(det) == 0.f) {
        return INFINITY;
    }

    glm::vec3 tv = r.origin - v0;
    glm::vec3 qv = glm::cross(tv, e0);

    bary.y = glm::dot(tv, pv) / det;
    bary.z = glm::dot(r.direction, qv) / det;
    bary.x = 1.f - bary.y - bary.z;
    float t = glm::dot(e1, qv) / det;
    
    if (bary.x >= 0 && bary.y >= 0 && bary.z >= 0 && t >= 0) {
        return t;
    }
    else {
        return INFINITY;
    }
}

__host__ __device__ float SpherePdf(const LightGeom& light, glm::vec3 point) {
    glm::vec3 center = light.position;
    float radius = light.radius;
    glm::vec3 centerToRef = glm::normalize(point - center);
    float distCenter = glm::length(point - center);
    // If inside the sphere
    if (distCenter <= radius) {
        return 0.f;
    }
    // Compute general sphere PDF
    float sinThetaMax2 = (radius * radius) / (distCenter * distCenter);
    float cosThetaMax = sqrt(glm::max(0.0f, 1.0f - sinThetaMax2));
    return 1.f / (TWO_PI * (1 - cosThetaMax));
}



__host__ __device__ bool getAnyHit(
    const Ray& r,
    char curSchannel,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    char* vertexSchannel,
    float maxt)
{
    // first check light source
    for (int i = 0; i < lightgeoms_size; ++i) {
        LightGeom& light = lightgeoms[i];
        LightType type = light.type;
        glm::vec3 u = light.u;
        // face light
        if (type == RECTLIGHT) {
            glm::vec3 v = light.v;
            glm::vec3 normal = glm::normalize(glm::cross(u, v));
            //if (dot(normal, r.direction) > 0.f) // if we hit a light from back
            //    continue;
            u *= 1.0f / glm::dot(u, u);
            v *= 1.0f / glm::dot(v, v);
            glm::vec3 position = light.position;
            glm::vec4 plane = glm::vec4(normal, glm::dot(normal, position));

            float distance = RectIntersect(position, u, v, plane, r);
            if (distance > 0 && distance < maxt) {
                return true;
            }
        }
        // sphere light
        if (type == SPHERELIGHT || type == SPOTLIGHT) {
            glm::vec3 position = light.position;
            float radius = light.radius;
            float distance = SphereIntersect(radius, position, r);
            bool inAngle = true;
            if (type == SPOTLIGHT) {
                float anglecos = glm::dot(-r.direction, light.u);
                if (anglecos < cos(light.outerAngle))
                    inAngle = false;
            }
            if (distance > 0 && distance < maxt && inAngle) {
                return true;
            }
        }
    }

    // check meshes, if distance smaller than light source than override it
    glm::vec3 barycentricParameters;
    glm::ivec3 vertIds;
#if PT_USEBVH
    // traversal stack
    int stack[64];
    int ptr = 0;
    stack[ptr++] = -1;  // adding a tag to record if traversed without intersection
    int bvhIdx = 0;
    while (bvhIdx >= 0) 
    {
        LinearBVHNode curNode = bvhNodes[bvhIdx];
        if (curNode.nPrimitives > 0) {
            // leaf node
            Geom geom = geoms[curNode.geomID];
            float distance = -1;
            if (geom.type == TRIANGLE)
            {
                vertIds = geom.vertIds;
                char triSchannel = glm::min(vertexSchannel[vertIds[0]], glm::min(vertexSchannel[vertIds[1]], vertexSchannel[vertIds[2]]));
#if PT_SHADOW_CHANNEL
                if (!ChannelCheck[curSchannel][triSchannel]) {
                    distance = INFINITY;
                }
                else
#endif // PT_SHADOW_CHANNEL
                {
                    distance = TriangleIntersect(vertexPos[vertIds[0]], vertexPos[vertIds[1]], vertexPos[vertIds[2]], r, barycentricParameters);
                }
            }
            else if (geom.type == SPHERE)
            {
                distance = SphereIntersect(geom.radius, geom.center, r);
            }

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (distance > 0.0f && distance < maxt)
            {
                return true;
            }
        }
        else {
            int leftIndex = bvhIdx + 1;
            int rightIndex = curNode.secondChildOffset;
            LinearBVHNode& leftNode = bvhNodes[leftIndex];
            LinearBVHNode& rightNode = bvhNodes[rightIndex];
            float leftHit = 0.0, rightHit = 0.0;
            leftHit = AABBIntersect(leftNode.bounds.pMin, leftNode.bounds.pMax, r);
            rightHit = AABBIntersect(rightNode.bounds.pMin, rightNode.bounds.pMax, r);
            // chech hit and distance
            if (leftHit > 0.0 && rightHit > 0.0) {
                int nextIndex;
                if (leftHit > rightHit) {
                    bvhIdx = rightIndex;     // first go right
                    nextIndex = leftIndex;   // go left later
                }
                else {
                    bvhIdx = leftIndex;      // first go left
                    nextIndex = rightIndex;  // go right later
                }
                stack[ptr++] = nextIndex;
                continue;
            }
            else if (leftHit > 0.) {
                bvhIdx = leftIndex;
                continue;
            }
            else if (rightHit > 0.) {
                bvhIdx = rightIndex;
                continue;
            }
        }
        // why we are here ?
        // 1. after checking a leaf node of meshBVH, we must go back
        // 2. we miss all the childnode of current node, we must go back
        bvhIdx = stack[--ptr];
    }
#else
    for (int i = 0; i < geoms_size; i++)
    {
        Geom geom = geoms[i];
        float distance = -1;
        if (geom.type == TRIANGLE)
        {
            vertIds = geom.vertIds;
            char triSchannel = glm::min(vertexSchannel[vertIds[0]], glm::min(vertexSchannel[vertIds[1]], vertexSchannel[vertIds[2]]));
#if PT_SHADOW_CHANNEL
            if (!ChannelCheck[curSchannel][triSchannel]) {
                distance = INFINITY;
            }
            else
#endif // PT_SHADOW_CHANNEL
            {
                distance = TriangleIntersect(vertexPos[vertIds[0]], vertexPos[vertIds[1]], vertexPos[vertIds[2]], r, barycentricParameters);
            }
        }
        else if (geom.type == SPHERE)
        {
            distance = SphereIntersect(geom.radius, geom.center, r);
        }

        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (distance > 0.0f && distance < maxt)
        {
            return true;
        }
    }
#endif // PT_USEBVH
    // No intersections before maxt
    return false;
}


__host__ __device__ bool getClosestHit(
    const Ray& r,
    char curSchannel,
    int depth, 
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    glm::vec3* vertexNor,
    glm::vec4* vertexUV,
    char* vertexSchannel,
    Material* materials,
    ShadeableIntersection& intersection,
    float normalOffset,
    PTCullingOptions culling)
{
    float t = INFINITY;

    // first check light source
#if PT_HIDE_EMITTERS
    if (depth > 0)
#endif // PT_HIDE_EMITTERS
    for (int i = 0; i < lightgeoms_size; ++i) {
        LightGeom& light = lightgeoms[i];
        LightType type = light.type;
        glm::vec3 u = light.u;
        // face light
        if (type == RECTLIGHT) {
            glm::vec3 v = light.v;
            glm::vec3 uvcross = glm::cross(u, v);
            glm::vec3 normal = glm::normalize(uvcross);
            u *= 1.0f / glm::dot(u, u);
            v *= 1.0f / glm::dot(v, v);
            glm::vec3 position = light.position;
            glm::vec4 plane = glm::vec4(normal, glm::dot(normal, position));

            float distance = RectIntersect(position, u, v, plane, r);
            if (distance < t) {
                t = distance;
                float cosTheta = dot(-r.direction, normal);
                intersection.pdf_Li = (t * t) / (glm::length(uvcross) * cosTheta);
                intersection.lightEmission = light.emission;
                intersection.materialId = -1;
            }
        }
        // sphere light
        if (type == SPHERELIGHT || type == SPOTLIGHT) {
            glm::vec3 position = light.position;
            float radius = light.radius;
            float distance = SphereIntersect(radius, position, r);
            bool inAngle = true;
            float falloff = 1.f;
            if (type == SPOTLIGHT) {
                float anglecos = glm::dot(-r.direction, light.u);
                if (anglecos < cos(light.outerAngle))
                    inAngle = false;
                falloff = glm::smoothstep(cos(light.outerAngle), cos(light.innerAngle), glm::max(anglecos, 0.f));
            }
            if (distance < t && inAngle) {
                t = distance;
                glm::vec3 hitPoint = r.origin + t * r.direction;
                intersection.pdf_Li = SpherePdf(light, r.origin);
                intersection.lightEmission = light.emission * falloff;
                intersection.materialId = -1;
            }
        }
    }

    // check meshes, if distance smaller than light source than override it
    int hit_geom_index = -1;
    int materialID = -1;
    GeomType hitType;
    glm::vec3 barycentricParameters;
    glm::ivec3 vertIds;
    glm::vec3 vp0, vp1, vp2, center;
    char hitSchannel = 0;
#if PT_USEBVH
    // traversal stack
    int stack[64];
    int ptr = 0;
    stack[ptr++] = -1;  // adding a tag to record if traversed without intersection
    int bvhIdx = 0;
    while (bvhIdx >= 0)
    {
        LinearBVHNode curNode = bvhNodes[bvhIdx];
        if (curNode.nPrimitives > 0) {
            // leaf node
            Geom geom = geoms[curNode.geomID];
            float distance = -1;
            glm::vec3 temp_bary;
            glm::ivec3 temp_vertIds;
            glm::vec3 temp_vp0, temp_vp1, temp_vp2, temp_center;
            char triSchannel = 0;
            if (geom.type == TRIANGLE)
            {
                temp_vertIds = geom.vertIds;
                temp_vp0 = vertexPos[temp_vertIds[0]];
                temp_vp1 = vertexPos[temp_vertIds[1]];
                temp_vp2 = vertexPos[temp_vertIds[2]];
                // vertex offset
                if (normalOffset > 0.f) {
                    normalOffset = glm::min((float)PT_LINE_MAXWIDTH, normalOffset);
                    temp_vp0 += normalOffset * vertexNor[temp_vertIds[0]] * glm::length(temp_vp0 - r.origin);
                    temp_vp1 += normalOffset * vertexNor[temp_vertIds[1]] * glm::length(temp_vp1 - r.origin);
                    temp_vp2 += normalOffset * vertexNor[temp_vertIds[2]] * glm::length(temp_vp2 - r.origin);
                }
                // shadow channel
                triSchannel = glm::min(vertexSchannel[temp_vertIds[0]], glm::min(vertexSchannel[temp_vertIds[1]], vertexSchannel[temp_vertIds[2]]));
#if PT_SHADOW_CHANNEL
                if (!ChannelCheck[curSchannel][triSchannel]) {
                    distance = INFINITY;
                }
                else
#endif // PT_SHADOW_CHANNEL
                {
                    distance = TriangleIntersect(temp_vp0, temp_vp1, temp_vp2, r, temp_bary);
                }
            }
            else if (geom.type == SPHERE)
            {
                temp_center = geom.center;
                distance = SphereIntersect(geom.radius + glm::min((float)PT_LINE_MAXWIDTH, normalOffset) * glm::length(temp_center - r.origin), temp_center, r);
            }
            // Culling
            bool cull = false;
            if (culling != PTCullingOptions::CULLNONE) {
                glm::vec3 tempnormal;
                if (geom.type == TRIANGLE) {
                    tempnormal = glm::normalize(vertexNor[temp_vertIds[0]] * temp_bary.x + vertexNor[temp_vertIds[1]] * temp_bary.y + vertexNor[temp_vertIds[2]] * temp_bary.z);
                }
                else if (geom.type == SPHERE) {
                    tempnormal = glm::normalize(r.origin + distance * r.direction - temp_center);
                }
                if (culling == PTCullingOptions::CULLFRONT) {
                    cull = glm::dot(-r.direction, tempnormal) > 0.f;
                }
                else if (culling == PTCullingOptions::CULLBACK) {
                    cull = glm::dot(-r.direction, tempnormal) < 0.f;
                }
            }
            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (distance > 0.0f && distance < t &&!cull)
            {
                t = distance;
                hit_geom_index = curNode.geomID;
                materialID = geom.materialid;
                hitType = geom.type;
                // copy some hit info
                vertIds = temp_vertIds;
                barycentricParameters = temp_bary;
                vp0 = temp_vp0;
                vp1 = temp_vp1;
                vp2 = temp_vp2;
                center = temp_center;
                hitSchannel = triSchannel;
            }
        }
        else {
            int leftIndex = bvhIdx + 1;
            int rightIndex = curNode.secondChildOffset;
            LinearBVHNode& leftNode = bvhNodes[leftIndex];
            LinearBVHNode& rightNode = bvhNodes[rightIndex];
            float leftHit = 0.0, rightHit = 0.0;
            leftHit = AABBIntersect(leftNode.bounds.pMin, leftNode.bounds.pMax, r);
            rightHit = AABBIntersect(rightNode.bounds.pMin, rightNode.bounds.pMax, r);
            // chech hit and distance
            if (leftHit > 0.0 && rightHit > 0.0) {
                int nextIndex;
                if (leftHit > rightHit) {
                    bvhIdx = rightIndex;     // first go right
                    nextIndex = leftIndex;   // go left later
                }
                else {
                    bvhIdx = leftIndex;      // first go left
                    nextIndex = rightIndex;  // go right later
                }
                stack[ptr++] = nextIndex;
                continue;
            }
            else if (leftHit > 0.) {
                bvhIdx = leftIndex;
                continue;
            }
            else if (rightHit > 0.) {
                bvhIdx = rightIndex;
                continue;
            }
        }
        // why we are here ?
        // 1. after checking a leaf node of meshBVH, we must go back
        // 2. we miss all the childnode of current node, we must go back
        bvhIdx = stack[--ptr];
    }
#else
    for (int i = 0; i < geoms_size; i++)
    {
        Geom geom = geoms[i];
        float distance = -1;
        glm::vec3 temp_bary;
        glm::ivec3 temp_vertIds;
        glm::vec3 temp_vp0, temp_vp1, temp_vp2, temp_center;
        char triSchannel = 0;
        if (geom.type == TRIANGLE)
        {
            temp_vertIds = geom.vertIds;
            temp_vp0 = vertexPos[temp_vertIds[0]];
            temp_vp1 = vertexPos[temp_vertIds[1]];
            temp_vp2 = vertexPos[temp_vertIds[2]];
            // vertex offset
            if (normalOffset > 0.f) {
                normalOffset = glm::min((float)PT_LINE_MAXWIDTH, normalOffset);
                temp_vp0 += normalOffset * vertexNor[temp_vertIds[0]] * glm::length(temp_vp0 - r.origin);
                temp_vp1 += normalOffset * vertexNor[temp_vertIds[1]] * glm::length(temp_vp1 - r.origin);
                temp_vp2 += normalOffset * vertexNor[temp_vertIds[2]] * glm::length(temp_vp2 - r.origin);
            }
            // shadow channel
            triSchannel = glm::min(vertexSchannel[temp_vertIds[0]], glm::min(vertexSchannel[temp_vertIds[1]], vertexSchannel[temp_vertIds[2]]));
#if PT_SHADOW_CHANNEL
            if (!ChannelCheck[curSchannel][triSchannel]) {
                distance = INFINITY;
            }
            else
#endif // PT_SHADOW_CHANNEL
            {
                distance = TriangleIntersect(temp_vp0, temp_vp1, temp_vp2, r, temp_bary);
            }
        }
        else if (geom.type == SPHERE)
        {
            temp_center = geom.center;
            distance = SphereIntersect(geom.radius + glm::min((float)PT_LINE_MAXWIDTH, normalOffset) * glm::length(temp_center - r.origin), temp_center, r);
        }
        // Culling
        bool cull = false;
        if (culling != PTCullingOptions::CULLNONE) {
            glm::vec3 tempnormal;
            if (geom.type == TRIANGLE) {
                tempnormal = glm::normalize(vertexNor[temp_vertIds[0]] * temp_bary.x + vertexNor[temp_vertIds[1]] * temp_bary.y + vertexNor[temp_vertIds[2]] * temp_bary.z);
            }
            else if (geom.type == SPHERE) {
                tempnormal = glm::normalize(r.origin + distance * r.direction - temp_center);
            }
            if (culling == PTCullingOptions::CULLFRONT) {
                cull = glm::dot(-r.direction, tempnormal) > 0.f;
            }
            else if (culling == PTCullingOptions::CULLBACK) {
                cull = glm::dot(-r.direction, tempnormal) < 0.f;
            }
        }
        // Compute the minimum t from the intersection tests to determine what
        // scene geometry object was hit first.
        if (distance > 0.0f && distance < t && !cull)
        {
            t = distance;
            hit_geom_index = i;
            materialID = geom.materialid;
            hitType = geom.type;
            // copy some hit info
            vertIds = temp_vertIds;
            barycentricParameters = temp_bary;
            vp0 = temp_vp0;
            vp1 = temp_vp1;
            vp2 = temp_vp2;
            center = temp_center;
            hitSchannel = triSchannel;
        }
    }
#endif // PT_USEBVH

    // No intersections
    if (t == INFINITY)
        return false;

    // hit
    intersection.t = t;
    glm::vec3 hitPos = r.origin + r.direction * t;

    // Ray hit a triangle and not a light source
    if (hit_geom_index != -1) {
        intersection.materialId = materialID;

        if (hitType == SPHERE) {

            intersection.surfaceNormal = glm::normalize(hitPos - center);
            // no texture support
        }
        else {
            // Normals
            glm::vec3 vn0 = vertexNor[vertIds[0]];
            glm::vec3 vn1 = vertexNor[vertIds[1]];
            glm::vec3 vn2 = vertexNor[vertIds[2]];

            // TexCoords
            glm::vec4 tc0_01 = vertexUV[vertIds[0]];
            glm::vec4 tc1_01 = vertexUV[vertIds[1]];
            glm::vec4 tc2_01 = vertexUV[vertIds[2]];
            glm::vec2 tc0_0 = glm::vec2(tc0_01.x, tc0_01.y);
            glm::vec2 tc1_0 = glm::vec2(tc1_01.x, tc1_01.y);
            glm::vec2 tc2_0 = glm::vec2(tc2_01.x, tc2_01.y);
            glm::vec2 tc0_1 = glm::vec2(tc0_01.z, tc0_01.w);
            glm::vec2 tc1_1 = glm::vec2(tc1_01.z, tc1_01.w);
            glm::vec2 tc2_1 = glm::vec2(tc2_01.z, tc2_01.w);

            // Interpolate texture coords and normals using barycentric coords
            intersection.texCoord[0] = tc0_0 * barycentricParameters.x + tc1_0 * barycentricParameters.y + tc2_0 * barycentricParameters.z;
            intersection.texCoord[1] = tc0_1 * barycentricParameters.x + tc1_1 * barycentricParameters.y + tc2_1 * barycentricParameters.z;
            // Interpolate normals
            intersection.surfaceNormal = glm::normalize(vn0 * barycentricParameters.x + vn1 * barycentricParameters.y + vn2 * barycentricParameters.z);

            // Calculate tangent (calculate bitangent = cross(normal, tangent) later in shading kernel)
            glm::vec3 deltaPos1 = vp1 - vp0;
            glm::vec3 deltaPos2 = vp2 - vp0;

            int normalUVChannel = materials[materialID].normalmapTexUV;
            glm::vec2 deltaUV1 = normalUVChannel == 1 ? (tc1_1 - tc0_1) : (tc1_0 - tc0_0);
            glm::vec2 deltaUV2 = normalUVChannel == 1 ? (tc2_1 - tc0_1) : (tc2_0 - tc0_0);

            float invdet = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
            glm::vec3 tangent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * invdet;

            intersection.tangent = glm::normalize(tangent - intersection.surfaceNormal * glm::dot(intersection.surfaceNormal, tangent));
        }

        intersection.schannel = hitSchannel;
    }
    return true;
}