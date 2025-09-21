#include "interactions.h"
#include "intersections.h"
#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 cosineSampleHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(fmax(0.f, 1 - up * up)); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ glm::vec3 uniformSampleHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = u01(rng);
    float over = sqrt(fmax(0.f, 1 - up * up));
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}







__host__ __device__ glm::vec3 F_Diffuse(
    const PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 wiW,
    const Material& m,
    float& pdf)
{
    pdf = glm::dot(wiW, normal) * INV_PI;
    return m.color * INV_PI * abs(glm::dot(wiW, normal));
}

__host__ __device__ glm::vec3 F_Specular(
    const PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 wiW,
    const Material& m,
    float& pdf)
{
    pdf = 0.f;
    return glm::vec3(0.f);
}

__host__ __device__ glm::vec3 F_Disney(
    const PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 wiW,
    const Material& m,
    float& pdf)
{
    // TODO
    pdf = glm::dot(wiW, normal) * INV_PI;
    return m.color;
}





__host__ __device__ void Sample_f_Diffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 wiW = cosineSampleHemisphere(normal, rng);

    //float absdot = abs(glm::dot(wiW, normal));
    //glm::vec3 bsdf = m.color * INV_PI;
    //float pdf = glm::dot(wiW, normal) * INV_PI;
    //pathSegment.color *= absdot * bsdf / pdf;

    pathSegment.throughput *= m.color; // absdot * bsdf / pdf = albedo
    pathSegment.pdf = glm::dot(wiW, normal) * INV_PI;
    pathSegment.ray = { intersect + wiW * EPSILON, wiW };
}

__host__ __device__ void Sample_f_Specular(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec3& woW = -pathSegment.ray.direction;
    // sample
    const float VdotN = glm::dot(woW, normal);
    const float eta = VdotN > 0.f ? 1.f / m.ior : m.ior;
    const float F_wo = dielectricFresnel(abs(VdotN), eta);
    float refractProb = m.transmission * (1.f - F_wo);

    glm::vec3 wiW;
    bool refract = false;
    if (u01(rng) < refractProb) {
        wiW = glm::refract(-woW, VdotN > 0.f ? normal : -normal, eta);
        refract = true;
    }
    else {
        wiW = glm::reflect(-woW, normal);
    }
    // evaluate
    if (refract) {
        pathSegment.throughput *= m.color * eta * eta;
    }
    else {
        pathSegment.throughput *= m.color;
    }
    pathSegment.pdf = INFINITY; // infinity
    pathSegment.ray = { intersect + wiW * EPSILON, wiW };
}

__host__ __device__ void Sample_f_Disney(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    // TODO
    glm::vec3 wiW = cosineSampleHemisphere(normal, rng);

    //float absdot = abs(glm::dot(wiW, normal));
    //glm::vec3 bsdf = m.color * INV_PI;
    //float pdf = glm::dot(wiW, normal) * INV_PI;
    //pathSegment.color *= absdot * bsdf / pdf;

    pathSegment.throughput *= m.color; // absdot * bsdf / pdf = albedo
    pathSegment.pdf = glm::dot(wiW, normal) * INV_PI;
    pathSegment.ray = { intersect + wiW * EPSILON, wiW };
}





__host__ __device__ void Sample_Li_Sphere(
    const LightGeom& light, 
    glm::vec3 scatterPos, 
    glm::vec3& lightDir, 
    glm::vec3& lightNor, 
    float& lightDist,
    float& pdf, 
    thrust::default_random_engine& rng) 
{
    glm::vec3 position = light.position;
    glm::vec3 sphereCentertoSurface = normalize(scatterPos - position);
    glm::vec3 sampledDir = uniformSampleHemisphere(sphereCentertoSurface, rng);

    float radius = light.radius;
    glm::vec3 lightSurfacePos = position + sampledDir * radius;

    glm::vec3 direction = lightSurfacePos - scatterPos;
    lightDist = glm::length(direction);
    float distSq = lightDist * lightDist;

    lightDir = direction / lightDist;
    lightNor = normalize(lightSurfacePos - position);
    pdf = distSq / ((PI * radius * radius) * 0.5 * abs(glm::dot(lightNor, lightDir)));
}

__host__ __device__ void Sample_Li_Rect(
    const LightGeom& light,
    glm::vec3 scatterPos,
    glm::vec3& lightDir,
    glm::vec3& lightNor,
    float& lightDist,
    float& pdf,
    thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r1 = u01(rng), r2 = u01(rng);

    glm::vec3 lightSurfacePos = light.position + light.u * r1 + light.v * r2;
    lightDir = lightSurfacePos - scatterPos;
    lightDist = glm::length(lightDir);
    float distSq = lightDist * lightDist;
    lightDir /= lightDist;
    glm::vec3 uvcross = glm::cross(light.u, light.v);
    lightNor = glm::normalize(uvcross);
    pdf = distSq / (glm::length(uvcross) * abs(glm::dot(lightNor, lightDir)));
}

__host__ __device__ void Sample_Li_Directional (
    const LightGeom& light,
    glm::vec3 scatterPos,
    glm::vec3& lightDir,
    glm::vec3& lightNor,
    float& lightDist,
    float& pdf,
    thrust::default_random_engine& rng)
{
    // position is actually direction
    lightDir = -normalize(light.position);
    lightNor = lightDir;
    lightDist = INFINITY;
    pdf = 1.0;
}






__host__ __device__ glm::vec3 Evaluate_f(
    const PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec3 wiW,
    const Material& m,
    float& pdf)
{
    if (m.type == DIFFUSE) {
        return F_Diffuse(pathSegment, intersect, normal, wiW, m, pdf);
    }
    else if (m.type == SPECULAR) {
        return F_Specular(pathSegment, intersect, normal, wiW, m, pdf);
    }
    else if (m.type == DISNEY) {
        return F_Disney(pathSegment, intersect, normal, wiW, m, pdf);
    }
    return glm::vec3(0.f);
}

__host__ __device__ void Sample_f(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    if (m.type == DIFFUSE) {
        Sample_f_Diffuse(pathSegment, intersect, normal, m, rng);
    }
    else if (m.type == SPECULAR) {
        Sample_f_Specular(pathSegment, intersect, normal, m, rng);
    }
    else if (m.type == DISNEY) {
        Sample_f_Disney(pathSegment, intersect, normal, m, rng);
    }
    
}

__host__ __device__ void Sample_Li(
    const LightGeom& light,
    glm::vec3 scatterPos,
    glm::vec3& lightDir,
    glm::vec3& lightNor,
    float& lightDist,
    float& pdf,
    thrust::default_random_engine& rng)
{
    if (light.type == SPHERELIGHT)
        Sample_Li_Sphere(light, scatterPos, lightDir, lightNor, lightDist, pdf, rng);
    else if (light.type == RECTLIGHT)
        Sample_Li_Rect(light, scatterPos, lightDir, lightNor, lightDist, pdf, rng);
    else if (light.type == DIRECTIONALLIGHT)
        Sample_Li_Directional(light, scatterPos, lightDir, lightNor, lightDist, pdf, rng);
}






__host__ __device__ void directLight(
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 radiance(0.f);
    glm::vec3 scatterPos = intersect + normal * EPSILON;

    if (lightgeoms_size == 0 || m.type == SPECULAR) { // No sample to light for perfectly specular material
        return;
    }

    //Pick a light to sample
    thrust::uniform_real_distribution<float> u01(0, 1);
    int index = int(u01(rng) * float(lightgeoms_size));

    glm::vec3 lightDir, lightNor;
    float lightDist, pdf_Li;
    LightGeom& light = lightgeoms[index];
    Sample_Li(light, scatterPos, lightDir, lightNor, lightDist, pdf_Li, rng);

    // check shadow
    Ray shadowRay = Ray(scatterPos, lightDir);
    bool inShadow = getAnyHit(shadowRay, geoms, geoms_size, lightgeoms, lightgeoms_size, vertexPos, lightDist - EPSILON);

    if (!inShadow) {
        float pdf_bsdf;
        glm::vec3 bsdf = Evaluate_f(pathSegment, intersect, normal, lightDir, m, pdf_bsdf);

        float misWeight = 1.0;
        if (light.type != DIRECTIONALLIGHT) // No MIS for directional light
            misWeight = powerHeuristic(pdf_Li, pdf_bsdf);

        if (pdf_bsdf > 0.0)
            radiance += misWeight * (light.emission * float(lightgeoms_size)) * bsdf / pdf_Li * pathSegment.throughput;
    }

    
    pathSegment.color += radiance;
}
