#pragma once

#include <cuda_runtime.h>

#include "glm/glm.hpp"

#include <string>
#include <vector>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType
{
    SPHERE,
    CUBE
};

// only for 4bit, stored in last 4 bit while sorting
// note that EMITTING and NONE_MAT(miss) will be tagged to be terminate
// so after sorting they should be at the right, and they should have a bit completely different from others in order to get remaining numbers 
enum MatType
{
    DIFFUSE,
    SPECULAR,
    PBR,
    // below are materials which are to "terminate" after this pass
    TER_DIFFUSE = 8,
    TER_SPECULAR,
    TER_PBR,
    LIGHT,
    NONE_MAT
};

struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom
{
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material
{
    MatType type;
    glm::vec3 color;
    float transmission;
    float ior;
    float roughness;
    float metallic;
    float emittance;
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
  glm::vec3 surfaceNormal;
  int materialId;
};
