#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    TRIANGLE
};

enum LightType
{
    SPHERELIGHT,
    RECTLIGHT,
    DIRECTIONALLIGHT
};

// only for 4bit, stored in last 4 bit while sorting
// note that EMITTING and NONE_MAT(miss) will be tagged to be terminate
// so after sorting they should be at the right, and they should have a bit completely different from others in order to get remaining numbers 
enum MatType
{
    DIFFUSE,
    SPECULAR,
    DISNEY,
    // below are materials which are to "terminate" after this pass
    TER_DIFFUSE = 8,
    TER_SPECULAR,
    TER_DISNEY,
    LIGHT,
    NONE_MAT
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;

    __host__ __device__ Ray(glm::vec3 ori, glm::vec3 dir) : origin(ori), direction(dir) {}

    __host__ __device__ Ray(const Ray& other) {
        std::memcpy(this, &other, sizeof(Ray));
    }

    __host__ __device__ Ray& operator=(const Ray& other) {
        if (this != &other) {
            std::memcpy(this, &other, sizeof(Ray));
        }
        return *this;
    }
};

struct Geom
{
    GeomType type;
    int materialid;
    union {
        glm::vec3 center;
        glm::ivec3 vertIds;
    };
    float radius;

    __host__ __device__ Geom(GeomType type) : type(type) {};

    __host__ __device__ Geom(const Geom& other) {
        std::memcpy(this, &other, sizeof(Geom));
    }

    __host__ __device__ Geom& operator=(const Geom& other) {
        if (this != &other) {
            std::memcpy(this, &other, sizeof(Geom));
        }
        return *this;
    }
};

struct LightGeom
{
    LightType type;
    glm::vec3 position;
    glm::vec3 emission;
    glm::vec3 u;
    glm::vec3 v;
    float radius;

    __host__ __device__ LightGeom(LightType type) : type(type) {}
};

struct Material
{
    MatType type;
    glm::vec3 color;
    glm::vec3 emission;
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float clearcoat;
    float coatroughness;
    float subsurface;

    __host__ __device__ Material() = default;

    __host__ __device__ Material(const Material& other) {
        std::memcpy(this, &other, sizeof(Material));
    }

    __host__ __device__ Material& operator=(const Material& other) {
        if (this != &other) {
            std::memcpy(this, &other, sizeof(Material));
        }
        return *this;
    }
};

struct Camera
{
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState
{
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment
{
    Ray ray;
    glm::vec3 color;
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
    float pdf;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection
{
    float t;
    int materialId;
    union {
        glm::vec3 surfaceNormal;
        glm::vec3 lightEmission;
    };
    union {
        glm::vec2 texCoord;
        float pdf_Li;
    };
    glm::vec3 tangent;

    __host__ __device__ ShadeableIntersection() : t(0), materialId(-1), surfaceNormal(0.f), texCoord(0.f), tangent(0.f) {};

    __host__ __device__ ShadeableIntersection(const ShadeableIntersection& other) {
        std::memcpy(this, &other, sizeof(ShadeableIntersection));
    }

    __host__ __device__ ShadeableIntersection& operator=(const ShadeableIntersection& other) {
        if (this != &other) {
            std::memcpy(this, &other, sizeof(ShadeableIntersection));
        }
        return *this;
    }
};
