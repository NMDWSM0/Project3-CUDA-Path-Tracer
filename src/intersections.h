#pragma once

#include "sceneStructs.h"
#include "bvh.h"

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a)
{
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t)
{
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v)
{
    return glm::vec3(m * v);
}

__host__ __device__ bool getClosestHit(
    const Ray& r,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    glm::vec3* vertexNor,
    glm::vec2* vertexUV,
    ShadeableIntersection& intersection);

__host__ __device__ bool getAnyHit(
    const Ray& r,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    float maxt);