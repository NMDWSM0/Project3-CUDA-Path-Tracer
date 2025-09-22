#pragma once
#include "utilities.h"

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/compatibility.hpp>

__host__ __device__ inline float srgbToLinear(float u) {
    u = glm::clamp(u, 0.0f, 1.0f);
    return (u <= 0.04045f) ? (u / 12.92f) : powf((u + 0.055f) / 1.055f, 2.4f);
}
__host__ __device__ inline glm::vec3 srgbToLinear(const glm::vec3& c) {
    return glm::vec3(srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b));
}

__host__ __device__ inline float linearToSrgb(float u) {
    u = glm::clamp(u, 0.0f, 1.0f);
    return (u <= 0.0031308f) ? (12.92f * u) : (1.055f * powf(u, 1.0f / 2.4f) - 0.055f);
}
__host__ __device__ inline glm::vec3 linearToSrgb(const glm::vec3& c) {
    return glm::vec3(linearToSrgb(c.r), linearToSrgb(c.g), linearToSrgb(c.b));
}

__host__ __device__ inline float luma709(const glm::vec3& c) {
    return 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
}

__host__ __device__ inline float safe_div(float a, float b) { return a / fmaxf(b, 1e-6f); }


__host__ __device__ inline glm::vec3 applyExposureEV(const glm::vec3& hdr, float exposureEV) {
    float k = exp2f(exposureEV);
    return hdr * k;
}

__host__ __device__ inline glm::vec3 applyWhiteBalance(const glm::vec3& c, float temperature, float tint) {
    float rGain = 1.0f + 0.10f * temperature - 0.05f * tint;
    float gGain = 1.0f - 0.02f * temperature + 0.10f * tint;
    float bGain = 1.0f - 0.10f * temperature - 0.05f * tint;
    return glm::vec3(c.r * rGain, c.g * gGain, c.b * bGain);
}

__host__ __device__ inline glm::vec3 applySaturation(const glm::vec3& c, float saturation) {
    float Y = luma709(c);
    return glm::mix(glm::vec3(Y), c, saturation);
}

__host__ __device__ inline glm::vec3 applyVibrance(const glm::vec3& c, float vibrance) {
    float Y = luma709(c);
    float sat = glm::length(c - glm::vec3(Y)) / (fmaxf(glm::length(c), 1e-5f));
    float t = (1.0f - sat);
    float k = 1.0f + vibrance * t; 
    float s = k; 
    return applySaturation(c, s);
}

__host__ __device__ inline glm::vec3 applyContrast(const glm::vec3& c, float contrast, float pivot = 0.18f) {
    return (c - glm::vec3(pivot)) * contrast + glm::vec3(pivot);
}

__host__ __device__ inline glm::vec3 applyASC_CDL(const glm::vec3& c,
    const glm::vec3& slope,
    const glm::vec3& offset,
    const glm::vec3& power_) {
    glm::vec3 x = c * slope + offset;
    x.r = powf(fmaxf(x.r, 0.0f), power_.r);
    x.g = powf(fmaxf(x.g, 0.0f), power_.g);
    x.b = powf(fmaxf(x.b, 0.0f), power_.b);
    return x;
}

__host__ __device__ inline glm::vec3 reinhardLuminancePreserving(const glm::vec3& hdr, float Lwhite = 0.0f) {
    float L = luma709(hdr);
    float Ld;
    if (Lwhite > 0.0f) {
        float Lw2 = Lwhite * Lwhite;
        Ld = (L * (1.0f + L / Lw2)) / (1.0f + L);
    }
    else {
        Ld = L / (1.0f + L);
    }
    float s = (L > 0.0f) ? (Ld / L) : 0.0f;
    return hdr * s;
}

__host__ __device__ inline glm::vec3 reinhardPerChannel(const glm::vec3& hdr, float exposure) {
    glm::vec3 m = hdr * exposure;
    return m / (glm::vec3(1.0f) + m);
}

__host__ __device__ inline glm::vec3 acesFitted(const glm::vec3& x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    glm::vec3 num = x * (a * x + b);
    glm::vec3 den = x * (c * x + d) + e;
    return glm::clamp(num / den, 0.0f, 1.0f);
}

struct ColorGradingParams {
    float exposureEV = 0.0f;   // Exposure(EV)
    float temperature = 0.0f;  // [-1, +1]
    float tint = 0.0f;         // [-1, +1]
    float saturation = 1.0f;   // 0..2
    float vibrance = 0.0f;     // 0..1 
    float contrast = 1.0f;     // 0..2 around pivot
    float contrastPivot = 0.18f;

    //tone curve
    bool useACES = true;       // ACES or Reinhard-L
    float reinhardLwhite = 0.0f; 

    glm::vec3 cdlSlope = glm::vec3(1.0f);
    glm::vec3 cdlOffset = glm::vec3(0.0f);
    glm::vec3 cdlPower = glm::vec3(1.0f);
};

__host__ __device__ inline glm::vec3 gradeAndToneMap(const glm::vec3& hdrLinear, const ColorGradingParams& P) {
    glm::vec3 c = hdrLinear;

    c = applyExposureEV(c, P.exposureEV);

    c = applyWhiteBalance(c, P.temperature, P.tint);

    c = applyASC_CDL(c, P.cdlSlope, P.cdlOffset, P.cdlPower);

    if (P.vibrance != 0.0f) c = applyVibrance(c, P.vibrance);
    c = applySaturation(c, P.saturation);

    c = applyContrast(c, P.contrast, P.contrastPivot);

    glm::vec3 sdr = P.useACES ? acesFitted(c) : glm::clamp(reinhardLuminancePreserving(c, P.reinhardLwhite), 0.0f, 1.0f);

    return linearToSrgb(sdr);
}
