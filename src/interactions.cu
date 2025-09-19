#include "interactions.h"

#include "utilities.h"

#include "bsdf.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
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

__host__ __device__ void Sample_f_diffuse(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    glm::vec3 wiW = calculateRandomDirectionInHemisphere(normal, rng);

    //float absdot = abs(glm::dot(wiW, normal));
    //glm::vec3 bsdf = m.color * INV_PI;
    //float pdf = glm::dot(wiW, normal) * INV_PI;
    //pathSegment.color *= absdot * bsdf / pdf;

    pathSegment.throughput *= m.color; // absdot * bsdf / pdf = albedo
    pathSegment.pdf = glm::dot(wiW, normal) * INV_PI;
    pathSegment.ray = { intersect + wiW * EPSILON, wiW };
}

__host__ __device__ void Sample_f_specular(
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
    pathSegment.pdf = 1.f;
    pathSegment.ray = { intersect + wiW * EPSILON * 20.f, wiW };
}

__host__ __device__ void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng)
{
    if (m.type == DIFFUSE) {
        Sample_f_diffuse(pathSegment, intersect, normal, m, rng);
    }
    else if (m.type == SPECULAR) {
        Sample_f_specular(pathSegment, intersect, normal, m, rng);
    }
    else if (m.type == PBR) {
        Sample_f_diffuse(pathSegment, intersect, normal, m, rng);
    }
}

