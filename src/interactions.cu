#include "interactions.h"
#include "intersections.h"
#include "utilities.h"
#include "postprocess.h"

#include <thrust/random.h>

#define TOON_SHADING 0

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

__host__ __device__ glm::vec3 sampleGTR1(
    float roughness, 
    glm::vec3 normal, 
    thrust::default_random_engine& rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r1 = u01(rng), r2 = u01(rng);

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

    float a = fmax(0.001f, roughness);
    float a2 = a * a;

    float phi = r1 * TWO_PI;

    float cosTheta = sqrt((1.f - pow(a2, 1.f - r2)) / (1.f - a2));
    float sinTheta = glm::clamp(sqrt(1.f - (cosTheta * cosTheta)), 0.f, 1.f);

    return cosTheta * normal
        + cos(phi) * sinTheta * perpendicularDirection1
        + sin(phi) * sinTheta * perpendicularDirection2;
}

__host__ __device__ glm::vec3 sampleGTR2(
    float roughness,
    glm::vec3 normal,
    thrust::default_random_engine& rng) 
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    float r1 = u01(rng), r2 = u01(rng);

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

    float a = fmax(0.001f, roughness);

    float phi = r1 * TWO_PI;

    float cosTheta = sqrt((1.f - r2) / (1.f + (a * a - 1.f) * r2));
    float sinTheta = glm::clamp(sqrt(1.f - (cosTheta * cosTheta)), 0.f, 1.f);
    float sinPhi = sin(phi);
    float cosPhi = cos(phi);

    return cosTheta * normal
        + cos(phi) * sinTheta * perpendicularDirection1
        + sin(phi) * sinTheta * perpendicularDirection2;
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
    pdf = 0.f;
    glm::vec3 bsdf(0.f);
    glm::vec3& woW = -pathSegment.ray.direction;
    const float NdotV = glm::dot(woW, normal);
    const float NdotL = glm::dot(wiW, normal);
    glm::vec3 ffnormal = NdotV > 0.f ? normal : -normal;
    const float eta = NdotV > 0.f ? 1.f / m.ior : m.ior;
    const float ffNdotV = glm::dot(woW, ffnormal);
    const float ffNdotL = glm::dot(wiW, ffnormal);

    glm::vec3 half;  // half vector
    if (ffNdotL > 0.f)
        half = glm::normalize(wiW + woW);
    else
        half = glm::normalize(wiW + woW * eta);
    if (glm::dot(half, ffnormal) < 0.f)
        half = -half;

    float F0 = (1.f - eta) / (1.f + eta); 
    F0 *= F0;

    // Model weights
    float dielectricWeight = (1.f - m.metallic) * (1.f - m.transmission);
    float metalWeight = m.metallic;
    float glassWeight = (1.f - m.metallic) * m.transmission;

    // Lobe probabilities
    float sWeight = fresnelSchlick(abs(NdotV));

    float diffPr = dielectricWeight * luminance(m.color);
    float dielectricPr = dielectricWeight * luminance(glm::mix(glm::vec3(F0), glm::vec3(1.f), sWeight));
    float metalPr = metalWeight * luminance(glm::mix(m.color, glm::vec3(1.f), sWeight));
    float glassPr = glassWeight;
    float clearCoatPr = 0.25f * m.clearcoat;

    // Normalize probabilities
    float invTotalPr = 1.f / (diffPr + dielectricPr + metalPr + glassPr + clearCoatPr);
    diffPr *= invTotalPr;
    dielectricPr *= invTotalPr;
    metalPr *= invTotalPr;
    glassPr *= invTotalPr;
    clearCoatPr *= invTotalPr;

    bool reflect = ffNdotL * ffNdotV > 0.f;

    float tmpPdf = 0.f;
    const float VDotH = abs(glm::dot(woW, half));
    // Diffuse
    if (diffPr > 0.f && reflect) {
        bsdf += evaluateDisneyDiffuse(m, woW, wiW, half, ffnormal, tmpPdf) * dielectricWeight;
        pdf += tmpPdf * diffPr;

#if TOON_SHADING
        if (abs(NdotL) > 0.01f) {
            bsdf /= abs(NdotL);
            bsdf *= (1 - pow(1 - abs(NdotL), 100.f)) * 0.5f;
        }
#endif
    }

    // Dielectric Reflection
    if (dielectricPr > 0.f && reflect) {
        // Normalize for interpolating based on Cspec0
        float F = dielectricFresnel(VDotH, 1.f / m.ior);

        bsdf += evaluateMicrofacetReflection(m, woW, wiW, half, ffnormal, glm::vec3(F), tmpPdf) * dielectricWeight;
        pdf += tmpPdf * dielectricPr;
    }

    // Metallic Reflection
    if (metalPr > 0.f && reflect) {
        // Tinted to base color
        glm::vec3 F = glm::mix(m.color, glm::vec3(1.f), fresnelSchlick(VDotH));

        bsdf += evaluateMicrofacetReflection(m, woW, wiW, half, ffnormal, F, tmpPdf) * metalWeight;
        pdf += tmpPdf * metalPr;
    }

    // Glass/Specular BSDF
    if (glassPr > 0.f) {
        // Dielectric fresnel
        float F = dielectricFresnel(VDotH, eta);
        if (reflect) {
            bsdf += evaluateMicrofacetReflection(m, woW, wiW, half, ffnormal, glm::vec3(F), tmpPdf) * glassWeight;
            pdf += tmpPdf * glassPr * F;
        }
        else {
            bsdf += evaluateMicrofacetRefraction(m, eta, woW, wiW, half, ffnormal, glm::vec3(F), tmpPdf) * glassWeight;
            pdf += tmpPdf * glassPr * (1.f - F);
        }
    }

    // Clearcoat
    if (clearCoatPr > 0.f && reflect) {
        bsdf += evaluateClearcoat(m, woW, wiW, half, ffnormal, tmpPdf) * 0.25f * m.clearcoat;
        pdf += tmpPdf * clearCoatPr;
    }

    return bsdf * abs(NdotL);  // bsdf * absdot
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
    thrust::uniform_real_distribution<float> u01(0, 1);
    const glm::vec3& woW = -pathSegment.ray.direction;
    const float NdotV = glm::dot(woW, normal);
    const float eta = NdotV > 0.f ? 1.f / m.ior : m.ior;
    glm::vec3 ffnormal = NdotV > 0.f ? normal : -normal;

    // Sample
    glm::vec3 wiW;
    float F0 = (1.f - eta) / (1.f + eta);
    F0 *= F0;

    // Model weights
    float dielectricWeight = (1.f - m.metallic) * (1.f - m.transmission);
    float metalWeight = m.metallic;
    float glassWeight = (1.f - m.metallic) * m.transmission;

    // Lobe probabilities
    float sWeight = fresnelSchlick(abs(NdotV));

    float diffPr = dielectricWeight * luminance(m.color);
    float dielectricPr = dielectricWeight * luminance(glm::mix(glm::vec3(F0), glm::vec3(1.f), sWeight));
    float metalPr = metalWeight * luminance(glm::mix(m.color, glm::vec3(1.f), sWeight));
    float glassPr = glassWeight;
    float clearCoatPr = 0.25f * m.clearcoat;

    // Normalize probabilities
    float invTotalPr = 1.f / (diffPr + dielectricPr + metalPr + glassPr + clearCoatPr);
    diffPr *= invTotalPr;
    dielectricPr *= invTotalPr;
    metalPr *= invTotalPr;
    glassPr *= invTotalPr;
    clearCoatPr *= invTotalPr;

    // CDF of the sampling probabilities
    float cdf[5];
    cdf[0] = diffPr;
    cdf[1] = cdf[0] + dielectricPr;
    cdf[2] = cdf[1] + metalPr;
    cdf[3] = cdf[2] + glassPr;
    cdf[4] = cdf[3] + clearCoatPr;

    float r1 = u01(rng) * cdf[4];
    glm::vec3 half;
    if (r1 < cdf[0]) {       // Diffuse
        wiW = cosineSampleHemisphere(ffnormal, rng);
    }
    else if (r1 < cdf[2]) {  // Dielectric + Metallic reflection
        half = sampleGTR2(m.roughness, ffnormal, rng);
        if (glm::dot(ffnormal, half) < 0.0)
            half = -half;
        wiW = glm::normalize(glm::reflect(-woW, half));
    }
    else if (r1 < cdf[3]) {  // Glass
        half = sampleGTR2(m.roughness, ffnormal, rng);
        float F = dielectricFresnel(abs(glm::dot(woW, half)), eta);
        if (glm::dot(ffnormal, half) < 0.0)
            half = -half;
        // Rescale random number for reuse
        r1 = (r1 - cdf[2]) / (cdf[3] - cdf[2]);
        if (r1 < F) { // Reflection
            wiW = glm::normalize(glm::reflect(-woW, half));
        }
        else {      // Transmission
            wiW = glm::normalize(glm::refract(-woW, half, eta));
        }
    }
    else { // Clearcoat
        half = sampleGTR1(m.coatroughness, ffnormal, rng);
        if (glm::dot(ffnormal, half) < 0.0)
            half = -half;
        wiW = glm::normalize(glm::reflect(-woW, half));
    }

    // Evaluate
    glm::vec3 bsdf(0.f);
    float pdf = 0.f;
    const float ffNdotL = glm::dot(wiW, ffnormal);
    const float ffNdotV = glm::dot(woW, ffnormal);

    bool reflect = ffNdotL * ffNdotV > 0.f;

    float tmpPdf = 0.f;
    const float VDotH = abs(glm::dot(woW, half));
    // Diffuse
    if (diffPr > 0.f && reflect) {
        bsdf += evaluateDisneyDiffuse(m, woW, wiW, half, ffnormal, tmpPdf) * dielectricWeight;
        pdf += tmpPdf * diffPr;

#if TOON_SHADING
        if (abs(ffNdotL) > 0.01f) {
            bsdf /= abs(ffNdotL);
            bsdf *= (1 - pow(1 - abs(ffNdotL), 100.f)) * 0.5f;
        }
#endif
    }

    // Dielectric Reflection
    if (dielectricPr > 0.f && reflect) {
        // Normalize for interpolating based on Cspec0
        float F = dielectricFresnel(VDotH, 1.f / m.ior);

        bsdf += evaluateMicrofacetReflection(m, woW, wiW, half, ffnormal, glm::vec3(F), tmpPdf) * dielectricWeight;
        pdf += tmpPdf * dielectricPr;
    }

    // Metallic Reflection
    if (metalPr > 0.f && reflect) {
        // Tinted to base color
        glm::vec3 F = glm::mix(m.color, glm::vec3(1.f), fresnelSchlick(VDotH));

        bsdf += evaluateMicrofacetReflection(m, woW, wiW, half, ffnormal, F, tmpPdf) * metalWeight;
        pdf += tmpPdf * metalPr;
    }

    // Glass/Specular BSDF
    if (glassPr > 0.f) {
        // Dielectric fresnel
        float F = dielectricFresnel(VDotH, eta);
        if (reflect) {
            bsdf += evaluateMicrofacetReflection(m, woW, wiW, half, ffnormal, glm::vec3(F), tmpPdf) * glassWeight;
            pdf += tmpPdf * glassPr * F;
        }
        else {
            bsdf += evaluateMicrofacetRefraction(m, eta, woW, wiW, half, ffnormal, glm::vec3(F), tmpPdf) * glassWeight;
            pdf += tmpPdf * glassPr * (1.f - F);
        }
    }

    // Clearcoat
    if (clearCoatPr > 0.f && reflect) {
        bsdf += evaluateClearcoat(m, woW, wiW, half, ffnormal, tmpPdf) * 0.25f * m.clearcoat;
        pdf += tmpPdf * clearCoatPr;
    }

    if (pdf > 0.f && !isnan(pdf)) {
        pathSegment.throughput *= bsdf * abs(ffNdotL) / pdf;
        pathSegment.pdf = pdf;
        pathSegment.ray = { intersect + wiW * EPSILON, wiW };
    }
    else {
        pathSegment.throughput = glm::vec3(0.f);
        pathSegment.pdf = 1.f;
        pathSegment.remainingBounces = 0;
    }
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




__device__ glm::vec3 Evaluate_EnvMap(Ray& r, cudaTextureObject_t envmapHandle)
{
    float theta = acos(glm::clamp(r.direction.y, -1.f, 1.f));
    glm::vec2 uv = glm::vec2((PI + glm::atan(r.direction.z, r.direction.x)) * INV_TWO_PI, theta * INV_PI);

    float4 c = tex2D<float4>(envmapHandle, uv.x, uv.y);

    return glm::vec3(c.x, c.y, c.z);
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
    LinearBVHNode* bvhNodes,
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
    bool inShadow = getAnyHit(shadowRay, bvhNodes, geoms, geoms_size, lightgeoms, lightgeoms_size, vertexPos, lightDist - EPSILON);

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


__device__ void getMatParams(
    Material& mat,
    const ShadeableIntersection& intersect,
    glm::vec3& normal,
    cudaTextureObject_t* textureHandles)
{
    // parse textures
    glm::vec2 uv = intersect.texCoord;
    if (mat.baseColorTexId >= 0) {
        float4 c = tex2D<float4>(textureHandles[mat.baseColorTexId], uv.x, uv.y);
        mat.color = srgbToLinear(glm::vec3(c.x, c.y, c.z));
    }
    if (mat.metallicRoughnessTexId >= 0) {
        float4 matrgh = tex2D<float4>(textureHandles[mat.metallicRoughnessTexId], uv.x, uv.y);
        mat.metallic = matrgh.x;
        mat.roughness = glm::max(matrgh.y, 0.001f);
    }
    if (mat.normalmapTexId >= 0) {
        float4 c = tex2D<float4>(textureHandles[mat.normalmapTexId], uv.x, uv.y);
        glm::vec3 normal_tspace = glm::normalize(glm::vec3(c.x, c.y, c.z));
        glm::vec3 bitangent = glm::cross(intersect.surfaceNormal, intersect.tangent);
        normal = glm::normalize(intersect.tangent * normal_tspace.x + bitangent * normal_tspace.y + intersect.surfaceNormal * normal_tspace.z);
    }
    if (mat.emissionmapTexId >= 0) {
        float4 c = tex2D<float4>(textureHandles[mat.emissionmapTexId], uv.x, uv.y);
        mat.emission = glm::vec3(c.x, c.y, c.z);
    }
}