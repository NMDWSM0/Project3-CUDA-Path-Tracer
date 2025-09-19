#pragma once

#include <glm/glm.hpp>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

__host__ __device__ float dielectricFresnel(float cosThetaI, float eta);