#include "bvh.h"
#include "defines.h"
// BVH build from, https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies

struct BVHPrimitiveInfo {
    BVHPrimitiveInfo() {}
    BVHPrimitiveInfo(size_t primitiveNumber, const Bounds3f& bounds)
        : primitiveNumber(primitiveNumber),
        bounds(bounds),
        centroid(.5f * bounds.pMin + .5f * bounds.pMax) {
    }
    size_t primitiveNumber;
    Bounds3f bounds;
    glm::vec3 centroid;
};

struct BVHBuildNode {
    // BVHBuildNode Public Methods
    void InitLeaf(int first, int n, const Bounds3f& b) {
        firstPrimOffset = first;
        nPrimitives = n;
        bounds = b;
        children[0] = children[1] = nullptr;
    }
    void InitInterior(int axis, BVHBuildNode* c0, BVHBuildNode* c1) {
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        splitAxis = axis;
        nPrimitives = 0;
    }
    Bounds3f bounds;
    BVHBuildNode* children[2];
    int splitAxis, firstPrimOffset, nPrimitives;
};

// BVHAccel Method Definitions
BVHAccel::BVHAccel(std::vector<std::shared_ptr<Primitive>>& p, int maxPrimsInNode)
    : maxPrimsInNode(std::min(255, maxPrimsInNode)), primitives(std::move(p)) 
{
    if (primitives.empty()) 
        return;

    // Initialize memory for buildnode
    buildNodeMemory = malloc(sizeof(BVHBuildNode) * primitives.size() * 2);
    curBuildNodePos = 0;

    // Initialize _primitiveInfo_ array for primitives
    std::vector<BVHPrimitiveInfo> primitiveInfo(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
        primitiveInfo[i] = BVHPrimitiveInfo(i, primitives[i]->bounds);

    // Build BVH tree for primitives using _primitiveInfo_
    int totalNodes = 0;
    std::vector<std::shared_ptr<Primitive>> orderedPrims;
    orderedPrims.reserve(primitives.size());
    BVHBuildNode* root;
    root = recursiveBuild(primitiveInfo, 0, primitives.size(), &totalNodes, orderedPrims);
    primitives.swap(orderedPrims);
    primitiveInfo.resize(0);

#if BVH_PRINT_BUILD_INFO
    printf("BVH created with %d nodes for %d "
        "primitives (%.2f MB)\n",
        totalNodes, (int)primitives.size(),
        float(totalNodes * sizeof(BVHBuildNode)) /
        (1024.f * 1024.f));
#endif

    nodes = new LinearBVHNode[totalNodes];
    int offset = 0;
    flattenBVHTree(root, &offset);

    this->root = root;
    this->totalNodes = totalNodes;
}

Bounds3f BVHAccel::WorldBound() const {
    return nodes ? nodes[0].bounds : Bounds3f();
}

BVHBuildNode* BVHAccel::newBuildNode(int count) const {
    BVHBuildNode* cur = (BVHBuildNode*)buildNodeMemory + curBuildNodePos;
    memset(cur, 0, sizeof(BVHBuildNode) * count);
    curBuildNodePos += count;
    return cur;
}

struct BucketInfo {
    int count = 0;
    Bounds3f bounds;
};

BVHBuildNode* BVHAccel::recursiveBuild(std::vector<BVHPrimitiveInfo>& primitiveInfo, int start,
    int end, int* totalNodes,
    std::vector<std::shared_ptr<Primitive>>& orderedPrims) {
    // CHECK_NE(start, end);
    BVHBuildNode* node = newBuildNode(1);
    (*totalNodes)++;
    // Compute bounds of all primitives in BVH node
    Bounds3f bounds;
    for (int i = start; i < end; ++i)
        bounds = Union(bounds, primitiveInfo[i].bounds);
    int nPrimitives = end - start;
    if (nPrimitives == 1) {
        // Create leaf _BVHBuildNode_
        int firstPrimOffset = orderedPrims.size();
        for (int i = start; i < end; ++i) {
            int primNum = primitiveInfo[i].primitiveNumber;
            orderedPrims.push_back(primitives[primNum]);
        }
        node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
        return node;
    }
    else {
        // Compute bound of primitive centroids, choose split dimension _dim_
        Bounds3f centroidBounds;
        for (int i = start; i < end; ++i)
            centroidBounds = Union(centroidBounds, primitiveInfo[i].centroid);
        int dim = centroidBounds.MaximumExtent();

        // Partition primitives into two sets and build children
        int mid = (start + end) / 2;
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            // fall back to equal counts
            std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
                &primitiveInfo[end - 1] + 1,
                [dim](const BVHPrimitiveInfo& a,
                    const BVHPrimitiveInfo& b) {
                        return a.centroid[dim] < b.centroid[dim];
                });
        }
        // Partition primitives using approximate SAH
        else if (nPrimitives <= 2) {
            // Partition primitives into equally-sized subsets
            std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
                &primitiveInfo[end - 1] + 1,
                [dim](const BVHPrimitiveInfo& a,
                    const BVHPrimitiveInfo& b) {
                        return a.centroid[dim] <
                            b.centroid[dim];
                });
        }
        else {
            // Allocate _BucketInfo_ for SAH partition buckets
            constexpr int nBuckets = 12;
            BucketInfo buckets[nBuckets];

            // Initialize _BucketInfo_ for SAH partition buckets
            for (int i = start; i < end; ++i) {
                int b = nBuckets *
                    centroidBounds.Offset(
                        primitiveInfo[i].centroid)[dim];
                if (b == nBuckets) b = nBuckets - 1;
                buckets[b].count++;
                buckets[b].bounds =
                    Union(buckets[b].bounds, primitiveInfo[i].bounds);
            }

            // Compute costs for splitting after each bucket
            float cost[nBuckets - 1];
            for (int i = 0; i < nBuckets - 1; ++i) {
                Bounds3f b0, b1;
                int count0 = 0, count1 = 0;
                for (int j = 0; j <= i; ++j) {
                    b0 = Union(b0, buckets[j].bounds);
                    count0 += buckets[j].count;
                }
                for (int j = i + 1; j < nBuckets; ++j) {
                    b1 = Union(b1, buckets[j].bounds);
                    count1 += buckets[j].count;
                }
                cost[i] = 1 +
                    (count0 * b0.SurfaceArea() +
                        count1 * b1.SurfaceArea()) /
                    bounds.SurfaceArea();
            }

            // Find bucket to split at that minimizes SAH metric
            float minCost = cost[0];
            int minCostSplitBucket = 0;
            for (int i = 1; i < nBuckets - 1; ++i) {
                if (cost[i] < minCost) {
                    minCost = cost[i];
                    minCostSplitBucket = i;
                }
            }

            // Either create leaf or split primitives at selected SAH
            // bucket
            float leafCost = nPrimitives;
            if (nPrimitives > maxPrimsInNode || minCost < leafCost) {
                BVHPrimitiveInfo* pmid = std::partition(
                    &primitiveInfo[start], &primitiveInfo[end - 1] + 1,
                    [=](const BVHPrimitiveInfo& pi) {
                        int b = nBuckets *
                            centroidBounds.Offset(pi.centroid)[dim];
                        if (b == nBuckets) b = nBuckets - 1;
                        return b <= minCostSplitBucket;
                    });
                mid = pmid - &primitiveInfo[0];
            }
            else {
                // Create leaf _BVHBuildNode_
                int firstPrimOffset = orderedPrims.size();
                for (int i = start; i < end; ++i) {
                    int primNum = primitiveInfo[i].primitiveNumber;
                    orderedPrims.push_back(primitives[primNum]);
                }
                node->InitLeaf(firstPrimOffset, nPrimitives, bounds);
                return node;
            }

            if (mid <= start || mid >= end) {
                mid = glm::clamp(mid, start + 1, end - 1);
            }
        }
        node->InitInterior(dim,
            recursiveBuild(primitiveInfo, start, mid,
                totalNodes, orderedPrims),
            recursiveBuild(primitiveInfo, mid, end,
                totalNodes, orderedPrims));
    }
    return node;
}

 int BVHAccel::flattenBVHTree(BVHBuildNode *node, int *offset) {
     LinearBVHNode *linearNode = &nodes[*offset];
     linearNode->bounds = node->bounds;
     int myOffset = (*offset)++;
     if (node->nPrimitives > 0) {
         linearNode->geomID = primitives[node->firstPrimOffset]->geomID;
         linearNode->nPrimitives = node->nPrimitives;
     } else {
         // Create interior flattened BVH node
         linearNode->nPrimitives = 0;
         flattenBVHTree(node->children[0], offset);
         linearNode->secondChildOffset =
             flattenBVHTree(node->children[1], offset);
     }
     return myOffset;
 }

BVHAccel::~BVHAccel() { delete[] nodes; free(buildNodeMemory); }


std::shared_ptr<BVHAccel> CreateBVHAccelerator(std::vector<std::shared_ptr<Primitive>>& prims, int maxPrimsInNode) 
{
    return std::make_shared<BVHAccel>(std::move(prims), maxPrimsInNode);
}