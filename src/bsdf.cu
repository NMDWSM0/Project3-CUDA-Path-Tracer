#include "bsdf.h"
#include "utilities.h"

__host__ __device__ float powerHeuristic(float a, float b) {
    float t = a * a;
    float w = t / (b * b + t);
    return isnan(w) ? 0 : w;
}

__host__ __device__ float luminance(glm::vec3 c) {
    return 0.212671 * c.x + 0.715160 * c.y + 0.072169 * c.z;
}

__host__ __device__ float GTR1(float NDotH, float a) {
    if (a >= 1.0)
        return INV_PI;
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return (a2 - 1.0) / (PI * log(a2) * t);
}

__host__ __device__ float GTR2(float NDotH, float a) {
    float a2 = a * a;
    float t = 1.0 + (a2 - 1.0) * NDotH * NDotH;
    return a2 / (PI * t * t);
}

__host__ __device__ float geometrySmith(float NDotV, float alphaG) {
    float a = alphaG * alphaG;
    float b = NDotV * NDotV;
    return (2.0 * NDotV) / (NDotV + sqrt(a + b - a * b));
}

__host__ __device__ float fresnelSchlick(float u) {
    float m = glm::clamp(1.0 - u, 0.0, 1.0);
    float m2 = m * m;
    return m2 * m2 * m;
}

__host__ __device__ float dielectricFresnel(float cosThetaI, float eta) {
    float sinThetaTSq = eta * eta * (1.0f - cosThetaI * cosThetaI);

    // Total internal reflection
    if (sinThetaTSq > 1.0)
        return 1.0;

    float cosThetaT = glm::sqrt(glm::max(1.0 - sinThetaTSq, 0.0));

    float rs = (eta * cosThetaT - cosThetaI) / (eta * cosThetaT + cosThetaI);
    float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);

    return 0.5f * (rs * rs + rp * rp);
}





__host__ __device__ glm::vec3 evaluateDisneyDiffuse(const Material& mat, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, float& pdf)
{
    pdf = 0.f;

    float HDotL = glm::dot(H, L);
    float NdotV = glm::dot(N, V);
    float NdotL = glm::dot(N, L);

    if (NdotL <= 0.f)
        return glm::vec3(0.f);

    float Rr = 2.f * mat.roughness * HDotL * HDotL;

    // Diffuse
    float FL = fresnelSchlick(NdotL);
    float FV = fresnelSchlick(NdotV);
    float Fretro = Rr * (FL + FV + FL * FV * (Rr - 1.f));
    float Fd = (1.f - 0.5f * FL) * (1.f - 0.5f * FV);

    // Fake subsurface
    float Fss90 = 0.5f * Rr;
    float Fss = glm::mix(1.f, Fss90, FL) * glm::mix(1.f, Fss90, FV);
    float ss = 1.25f * (Fss * (1.f / (NdotL + NdotV) - 0.5f) + 0.5f);

    pdf = NdotL * INV_PI;
    return INV_PI * mat.color * glm::mix(Fd + Fretro, ss, mat.subsurface);
}

__host__ __device__ glm::vec3 evaluateMicrofacetReflection(const Material& mat, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, glm::vec3 F, float& pdf)
{
    pdf = 0.f;

    float NDotH = glm::dot(N, H);
    float NdotV = glm::dot(N, V);
    float NdotL = glm::dot(N, L);

    if (NdotL <= 0.f)
        return glm::vec3(0.f);

    float a = mat.roughness;

    float D = GTR2(NDotH, a);
    float G1 = geometrySmith(abs(NdotV), a);
    float G2 = G1 * geometrySmith(abs(NdotL), a);

    pdf = G1 * D / (4.f * NdotV);
    return F * D * G2 / (4.f * NdotL * NdotV)/* + evaluateMicrofacetMultiScatter(mat, V, L)*/;
}

__host__ __device__ glm::vec3 evaluateMicrofacetRefraction(const Material& mat, float eta, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, glm::vec3 F, float& pdf)
{
    pdf = 0.f;

    float NDotH = glm::dot(N, H);
    float NdotV = glm::dot(N, V);
    float NdotL = glm::dot(N, L);
    float LDotH = glm::dot(L, H);
    float VDotH = glm::dot(V, H);

    if (NdotL >= 0.f)
        return glm::vec3(0.f);

    float a = mat.roughness;

    float D = GTR2(NDotH, a);
    float G1 = geometrySmith(abs(NdotV), a);
    float G2 = G1 * geometrySmith(abs(NdotL), a);
    float denom = LDotH + VDotH * eta;
    denom *= denom;
    float eta2 = eta * eta;
    float jacobian = abs(LDotH) / denom;

    pdf = G1 * fmax(0.f, VDotH) * D * jacobian / NdotV;
    return pow(mat.color, glm::vec3(0.5f)) * (1.f - F) * D * G2 * abs(VDotH) * jacobian * eta2 / abs(NdotL * NdotV);
}

__host__ __device__ glm::vec3 evaluateClearcoat(const Material& mat, glm::vec3 V, glm::vec3 L, glm::vec3 H, glm::vec3 N, float& pdf)
{
    pdf = 0.f;

    float NDotH = glm::dot(N, H);
    float NdotV = glm::dot(N, V);
    float NdotL = glm::dot(N, L);
    float VDotH = dot(V, H);

    if (NdotL <= 0.f)
        return glm::vec3(0.f);

    float F = glm::mix(0.04f, 1.f, fresnelSchlick(VDotH));
    float D = GTR1(NDotH, mat.coatroughness);
    float G = geometrySmith(NdotL, 0.25) * geometrySmith(NdotV, 0.25);
    float jacobian = 1.0 / (4.0 * VDotH);

    pdf = D * NDotH * jacobian;
    return glm::vec3(F) * D * G;
}