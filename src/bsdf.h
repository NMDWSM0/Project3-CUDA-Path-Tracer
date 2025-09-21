#pragma once

#include <glm/glm.hpp>

#include "sceneStructs.h"

__host__ __device__ float powerHeuristic(float a, float b);

__host__ __device__ float dielectricFresnel(float cosThetaI, float eta);