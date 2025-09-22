#pragma once

#include <glm/glm.hpp>

#include "sceneStructs.h"

__host__ __device__ float powerHeuristic(float a, float b);

__host__ __device__ float luminance(glm::vec3 c);

__host__ __device__ float GTR1(float NDotH, float a);

__host__ __device__ float GTR2(float NDotH, float a);

__host__ __device__ float geometrySmith(float NDotV, float alphaG);

__host__ __device__ float fresnelSchlick(float u);

__host__ __device__ float dielectricFresnel(float cosThetaI, float eta);



__host__ __device__ glm::vec3 evaluateDisneyDiffuse(const Material& mat, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, float& pdf);

__host__ __device__ glm::vec3 evaluateMicrofacetReflection(const Material& mat, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, glm::vec3 F, float& pdf);

__host__ __device__ glm::vec3 evaluateMicrofacetRefraction(const Material& mat, float eta, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, glm::vec3 F, float& pdf);

__host__ __device__ glm::vec3 evaluateClearcoat(const Material& mat, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, float& pdf);