#pragma once

#include "utilities.h"

class Bounds3f
{
public:
    glm::vec3 pMin, pMax;

    Bounds3f() : pMin(std::numeric_limits<float>::max()), pMax(std::numeric_limits<float>::lowest())
    {}

    explicit Bounds3f(const glm::vec3& p) : pMin(p), pMax(p) {}

    Bounds3f(const glm::vec3& p1, const glm::vec3& p2) :
        pMin(glm::min(p1, p2)),
        pMax(glm::max(p1, p2)) 
    {}

    bool operator==(const Bounds3f& b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }

    bool operator!=(const Bounds3f& b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }

    glm::vec3 Diagonal() const { return pMax - pMin; }

    float SurfaceArea() const {
        glm::vec3 d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

    float Volume() const {
        glm::vec3 d = Diagonal();
        return d.x * d.y * d.z;
    }

    int MaximumExtent() const {
        glm::vec3 d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }

    glm::vec3 Offset(const glm::vec3& p) const {
        glm::vec3 o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
};


inline Bounds3f Union(const Bounds3f& b1, const Bounds3f& b2) {
    return Bounds3f(glm::min(b1.pMin, b2.pMin),
        glm::max(b1.pMax, b2.pMax));
}

inline Bounds3f Union(const Bounds3f& b, const glm::vec3& p) {
    Bounds3f ret;
    ret.pMin = glm::min(b.pMin, p);
    ret.pMax = glm::max(b.pMax, p);
    return ret;
}

class Primitive
{
public:
    int geomID;
    Bounds3f bounds;
public:
    Primitive(int id) :
        geomID(id)
    {}

    Primitive(int id, const Bounds3f& bounds) :
        geomID(id), bounds(bounds)
    {}

    Primitive(int id, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) :
        geomID(id), bounds(Union(Bounds3f(p1, p2), p3))
    {}

    Primitive(int id, glm::vec3 center, float radius) :
        geomID(id), bounds(center - glm::vec3(radius), center + glm::vec3(radius))
    {}

};

// BVH, https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies

struct BVHBuildNode;

// BVHAccel Forward Declarations
struct BVHPrimitiveInfo;

struct LinearBVHNode {
    Bounds3f bounds;
    union {
        int geomID;   // leaf
        int secondChildOffset;  // interior
    };
    int nPrimitives;
};

// BVHAccel Declarations
class BVHAccel
{
public:

    // BVHAccel Public Methods
    BVHAccel(std::vector<std::shared_ptr<Primitive>>& p, int maxPrimsInNode = 1);
    Bounds3f WorldBound() const;
    ~BVHAccel();

private:
    // BVHAccel Private Methods
    BVHBuildNode* recursiveBuild(std::vector<BVHPrimitiveInfo>& primitiveInfo,
        int start, int end, int* totalNodes,
        std::vector<std::shared_ptr<Primitive>>& orderedPrims);

     int flattenBVHTree(BVHBuildNode *node, int *offset);

private:
    void* buildNodeMemory;
    mutable int curBuildNodePos;
    BVHBuildNode* newBuildNode(int count) const;

    // BVHAccel Data
public:
    const int maxPrimsInNode;
    std::vector<std::shared_ptr<Primitive>> primitives;
    LinearBVHNode* nodes = nullptr;
    BVHBuildNode* root = nullptr;
    int totalNodes;
};

std::shared_ptr<BVHAccel> CreateBVHAccelerator(std::vector<std::shared_ptr<Primitive>>& prims, int maxPrimsInNode);
