#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <OpenImageDenoise/oidn.hpp>

#include "defines.h"
#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "postprocess.h"
#include "../stream_compaction/efficient.h"

#define PACKINDEX(index, mat) (((index) << 4 ) | ((mat) & 0xF))
#define UNPACKINDEX(packed) ((packed) >> 4)
#define UNPACKMAT(packed) ((packed) & 0xF)

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if PT_ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // PT_ERRORCHECK
}

// scene
static Scene* hst_scene = nullptr;
static GuiDataContainer* guiData = nullptr;
// image & gbuffers
static glm::vec3* dev_image = nullptr;
static glm::vec3* dev_denoiseimage = nullptr;
static glm::vec3* dev_postimage = nullptr;
static glm::vec3* dev_GB_position = nullptr;
static glm::vec3* dev_GB_albedo = nullptr;  // in linear space
static glm::vec3* dev_GB_normal = nullptr;
static RenderLine* dev_lines = nullptr;
// geoms & bvhs
static Geom* dev_geoms = nullptr;
static LightGeom* dev_lightgeoms = nullptr;
static LinearBVHNode* dev_bvhnodes = nullptr;
static LinearBVHNode* dev_extendbvhnodes = nullptr;
static glm::vec3* dev_vertPos = nullptr;
static glm::vec3* dev_vertNor = nullptr;
static glm::vec4* dev_vertUV = nullptr;
static char* dev_vertSchannel = nullptr;
// mats & textures
static Material* dev_materials = nullptr;
static char* dev_mattypes = nullptr;
static cudaTextureObject_t* dev_texurehandles = nullptr;
static cudaTextureObject_t envmaphandle;
// paths & isects
static PathSegment* dev_paths_A = nullptr;
static PathSegment* dev_paths_B = nullptr;
static ShadeableIntersection* dev_intersections = nullptr;
static int* dev_pathremains = nullptr;
static int* dev_pathindices = nullptr;
// oidn 
static int oidn_deviceID = 0;
static cudaStream_t oidn_stream = nullptr;
static OIDNFilter oidn_filter = nullptr;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    const int pixelcount_pot = 1 << ilog2ceil(pixelcount);

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_denoiseimage, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoiseimage, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_postimage, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_postimage, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("initialize images");

    cudaMalloc(&dev_GB_position, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_GB_position, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_GB_albedo, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_GB_albedo, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_GB_normal, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_GB_normal, 0, pixelcount * sizeof(glm::vec3));
    checkCUDAError("initialize GBuffers");

    cudaMalloc(&dev_lines, pixelcount * sizeof(RenderLine));
    cudaMemset(dev_lines, 0, pixelcount * sizeof(RenderLine));
    checkCUDAError("initialize lines");

    cudaMalloc(&dev_paths_A, pixelcount * sizeof(PathSegment));
    cudaMalloc(&dev_paths_B, pixelcount * sizeof(PathSegment));
    checkCUDAError("initialize PathSegment");

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    checkCUDAError("initialize geoms");

    cudaMalloc(&dev_lightgeoms, scene->lightgeoms.size() * sizeof(LightGeom));
    cudaMemcpy(dev_lightgeoms, scene->lightgeoms.data(), scene->lightgeoms.size() * sizeof(LightGeom), cudaMemcpyHostToDevice);
    checkCUDAError("initialize lights");

    cudaMalloc(&dev_bvhnodes, scene->bvhAccel->totalNodes * sizeof(LinearBVHNode));
    if (scene->bvhAccel->totalNodes > 0) {
        cudaMemcpy(dev_bvhnodes, scene->bvhAccel->nodes, scene->bvhAccel->totalNodes * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
    }
    checkCUDAError("initialize bvh");

#if PT_LINE_RENDER
    cudaMalloc(&dev_extendbvhnodes, scene->extend_bvhAccel->totalNodes * sizeof(LinearBVHNode));
    if (scene->extend_bvhAccel->totalNodes > 0) {
        cudaMemcpy(dev_extendbvhnodes, scene->extend_bvhAccel->nodes, scene->extend_bvhAccel->totalNodes * sizeof(LinearBVHNode), cudaMemcpyHostToDevice);
    }
    checkCUDAError("initialize extendbvh");
#endif // PT_LINE_RENDER

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    if (scene->materials.size() > 0) {
        cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
    }
    checkCUDAError("initialize materials");

    cudaMalloc(&dev_mattypes, scene->materials.size() * sizeof(char));
    std::vector<char> mattypes;
    for (auto& m : scene->materials) { mattypes.push_back((char)(m.type)); }
    if (scene->materials.size() > 0) {
        cudaMemcpy(dev_mattypes, mattypes.data(), scene->materials.size() * sizeof(char), cudaMemcpyHostToDevice);
    }
    checkCUDAError("initialize mattypes");

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    checkCUDAError("initialize intersections");

    cudaMalloc(&dev_vertPos, scene->vertPos.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_vertPos, scene->vertPos.data(), scene->vertPos.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_vertNor, scene->vertNor.size() * sizeof(glm::vec3));
    cudaMemcpy(dev_vertNor, scene->vertNor.data(), scene->vertNor.size() * sizeof(glm::vec3), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_vertUV, scene->vertUV.size() * sizeof(glm::vec4));
    cudaMemcpy(dev_vertUV, scene->vertUV.data(), scene->vertUV.size() * sizeof(glm::vec4), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_vertSchannel, scene->vertSchannel.size() * sizeof(char));
    cudaMemcpy(dev_vertSchannel, scene->vertSchannel.data(), scene->vertSchannel.size() * sizeof(char), cudaMemcpyHostToDevice);
    checkCUDAError("initialize vertex buffers");

    cudaMalloc(&dev_pathremains, pixelcount * sizeof(int));
    cudaMalloc(&dev_pathindices, pixelcount * sizeof(int));
    checkCUDAError("initialize compaction related buffers");

    envmaphandle = scene->envMap.loadToCuda();
    checkCUDAError("initialize envmap");

    cudaMalloc(&dev_texurehandles, scene->textures.size() * sizeof(cudaTextureObject_t));
    for (int i = 0; i < scene->textures.size(); ++i) {
        auto handle = scene->textures[i].loadToCuda();
        cudaMemcpy(dev_texurehandles + i, &handle, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    }
    checkCUDAError("initialize textures");

    // buffer size used for scan should be bigger
    StreamCompaction::EfficientSharedMem::initializeBuffers(pixelcount_pot);
    checkCUDAError("EfficientSharedMem::initializeBuffers");

    // init oidn
    OIDNDevice dev = oidnNewCUDADevice(&oidn_deviceID, &oidn_stream, 1);
    oidnCommitDevice(dev);
    oidn_filter = oidnNewFilter(dev, "RT");

    oidnSetSharedFilterImage(oidn_filter, "color",
        dev_image, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y,
        0,                                      /*byteOffset*/
        sizeof(glm::vec3),                      /*pixelByteStride*/
        sizeof(glm::vec3) * cam.resolution.x);  /*rowByteStride*/
    oidnSetSharedFilterImage(oidn_filter, "albedo", 
        dev_GB_albedo, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y,
        0,
        sizeof(glm::vec3),
        sizeof(glm::vec3) * cam.resolution.x);
    oidnSetSharedFilterImage(oidn_filter, "normal",
        dev_GB_normal, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y,
        0,
        sizeof(glm::vec3),
        sizeof(glm::vec3) * cam.resolution.x);
    oidnSetSharedFilterImage(oidn_filter, "output",
        dev_denoiseimage, OIDN_FORMAT_FLOAT3, cam.resolution.x, cam.resolution.y,
        0,
        sizeof(glm::vec3),
        sizeof(glm::vec3) * cam.resolution.x);
    // use linear HDR color to denoise
    oidnSetFilterBool(oidn_filter, "hdr", true);
#if PT_REALTIME_DENOISE
    oidnSetFilterInt(oidn_filter, "quality", OIDN_QUALITY_FAST);
#else
    oidnSetFilterInt(oidn_filter, "quality", OIDN_QUALITY_HIGH);
#endif // PT_REALTIME_DENOISE
    oidnCommitFilter(oidn_filter);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_denoiseimage);
    cudaFree(dev_postimage);
    cudaFree(dev_GB_position);
    cudaFree(dev_GB_albedo);
    cudaFree(dev_GB_normal);
    cudaFree(dev_lines);
    cudaFree(dev_paths_A);
    cudaFree(dev_paths_B);
    cudaFree(dev_geoms);
    cudaFree(dev_lightgeoms);
    cudaFree(dev_bvhnodes);
    cudaFree(dev_extendbvhnodes);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // clean up any extra device memory
    cudaFree(dev_vertPos);
    cudaFree(dev_vertNor);
    cudaFree(dev_vertUV);
    cudaFree(dev_vertSchannel);
    cudaFree(dev_pathremains);
    cudaFree(dev_pathindices);
    cudaFree(dev_mattypes);
    cudaFree(dev_texurehandles);

    for (int i = 0; i < hst_scene->textures.size(); ++i) {
        hst_scene->textures[i].FreeCudaSide();
    }

    StreamCompaction::EfficientSharedMem::freeBuffers();

    checkCUDAError("pathtraceFree");
}

void pathtraceClear()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoiseimage, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_postimage, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_GB_position, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_GB_albedo, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_GB_normal, 0, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_lines, 0, pixelcount * sizeof(RenderLine));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceClear");
}

bool pathtraceFocusOnPixel(int px, int py) {
    Camera& cam = hst_scene->state.camera;
    int offset = px + py * cam.resolution.x;
    glm::vec3 pos;
    cudaMemcpy(&pos, dev_GB_position + offset, sizeof(glm::vec3), cudaMemcpyDeviceToHost);
    if (glm::length(pos - cam.position) > 100000.f) {
        return false;
    }
    float distance = glm::dot(pos - cam.position, cam.view);
    std::cout << distance << std::endl;
    if (abs(cam.focalDistance - distance) < EPSILON) {
        return false;
    }
    cam.focalDistance = distance;
    return true;
}







__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image, glm::vec3* postimage, ColorGradingParams params)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::vec3 finalCol = gradeAndToneMap(pix, params);
        postimage[index] = finalCol;

        glm::ivec3 color;
        color.x = glm::clamp((int)(finalCol.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(finalCol.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(finalCol.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void sendImageToPBOVec4(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec4* image, glm::vec3* postimage, ColorGradingParams params)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec4 pix = image[index];
        glm::vec3 finalCol = glm::vec3(0.5f * pix + 0.5f) * pix.a;

        postimage[index] = finalCol;

        glm::ivec3 color;
        color.x = glm::clamp((int)(finalCol.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(finalCol.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(finalCol.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__device__ __constant__ float G3x3[9] = {
    1.f, 2.f, 1.f,
    2.f, 4.f, 2.f,
    1.f, 2.f, 1.f
};

__global__ void blurLinesR(glm::ivec2 resolution, RenderLine* lines)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x >= resolution.x || y >= resolution.y) return;

    const int width = resolution.x;
    const int height = resolution.y;
    const int index = x + y * width;

    float       rAcc = 0.0f;
    glm::vec3   cAcc = glm::vec3(0.0f);
    float       wAcc = 0.0f;

#pragma unroll
    for (int dy = -1; dy <= 1; ++dy) {
#pragma unroll
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                float w = G3x3[(dy + 1) * 3 + (dx + 1)];
                const RenderLine& s = lines[nx + ny * width];
                rAcc += w * s.alpha;
                cAcc += w * s.color;
                wAcc += w;
            }
        }
    }

    float invW = (wAcc > 0.0f) ? (1.0f / wAcc) : 0.0f;
    lines[index].alpha = rAcc * invW;
    lines[index].color = cAcc * invW;
}

inline __host__ __device__ bool worldToPixel(
    const Camera& cam,
    const glm::vec3& P,
    glm::ivec2& outPx)
{
    const glm::vec3 d = P - cam.position;
    const float z = glm::dot(d, cam.view);

    if (z <= 0.0f) return false;

    const float u = glm::dot(d, cam.right);
    const float v = glm::dot(d, cam.up);

    const float x_img = -(u / z);
    const float y_img = -(v / z);

    const float cx = 0.5f * (cam.resolution.x);
    const float cy = 0.5f * (cam.resolution.y);

    const float px = x_img / cam.pixelLength.x + cx;
    const float py = y_img / cam.pixelLength.y + cy;

    outPx = { glm::floor(px + 0.5f), glm::floor(py + 0.5f) };

    return (px >= -0.5f && px <= cam.resolution.x - 0.5f &&
        py >= -0.5f && py <= cam.resolution.y - 0.5f);
}

inline __host__ __device__ float estimateLodByDistance(float distance)
{
    constexpr float k = 0.05f;
    return log2(distance * k);
}

__global__ void generateGBufferRayFromCamera(Camera cam, PathSegment* pathSegments, glm::vec2 offset)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.throughput = glm::vec3(1.0f);
        segment.color = glm::vec3(0.f);
        segment.pixelIndex = index;
        segment.remainingBounces = 1;
        segment.schannel = 0;

        // gbuffer ray, no DOF applied, offset is input for 2xSSAA
        segment.ray.origin = cam.position;
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + offset.x + 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + offset.y + 0.5f)
        );
    }
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 114514);
        thrust::uniform_real_distribution<float> u01(0, 1);

        segment.throughput = glm::vec3(1.0f);
        segment.color = glm::vec3(0.f);
        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.schannel = 0;

        float aspectRatio = (float)cam.resolution.x / cam.resolution.y;
        float apertureR = cam.lenRadius;
#if !PT_DOF
        apertureR = 0.f;
#endif // !PT_DOF
#if PT_AA
        // antialiasing by jittering the ray
        float offsetX = u01(rng);
        float offsetY = u01(rng);
#endif // PT_AA
        float xn = ((float)(x + offsetX) / cam.resolution.x) * 2.f - 1.f;
        float yn = ((float)(y + offsetY) / cam.resolution.y) * 2.f - 1.f;
        float xs = xn * (aspectRatio * tan(0.5f * cam.fov.y * PI / 180));
        float ys = yn * (tan(0.5f * cam.fov.y * PI / 180));
        
        glm::vec3 dtocenter = glm::vec3(-xs, -ys, 1.f);
        glm::vec3 Pf = cam.focalDistance * dtocenter;        // the focus point on clear plane

        float rad = sqrtf(u01(rng)) * apertureR;
        float theta = TWO_PI * u01(rng);
        float lensSampleX = rad * cos(theta);
        float lensSampleY = rad * sin(theta);
        glm::vec3 Apoint(lensSampleX, lensSampleY, 0.f);     // radom point on aperture

        // local space origin and direction
        glm::vec3 origin = Apoint;
        glm::vec3 dir = glm::normalize(Pf - Apoint);

        segment.ray.origin = cam.position + origin.x * cam.right + origin.y * cam.up + origin.z * cam.view;
        segment.ray.direction = glm::normalize(dir.x * cam.right + dir.y * cam.up + dir.z * cam.view);
    }
}


#if PT_MATERIAL_SORT
__global__ void computeIntersections(
    Camera cam,
    int iter,
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    glm::vec3* vertexNor,
    glm::vec4* vertexUV,
    char* vertexSchannel,
    Material* materials,
    RenderLine* lines,
    char* mattypes,
    ShadeableIntersection* intersections,
    int* pathIndices)
#else
__global__ void computeIntersections(
    Camera cam,
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    glm::vec3* vertexNor,
    glm::vec2* vertexUV,
    char* vertexSchannel,
    Material* materials,
    RenderLine* lines,
    ShadeableIntersection* intersections)
#endif // PT_MATERIAL_SORT
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
#if PT_MATERIAL_SORT
#if PT_RUSSIAN_ROULETTE
        // compute rr prob eariler
        glm::vec3 throughput = pathSegments[path_index].throughput;
        float q = fminf(fmaxf(throughput.r, fmaxf(throughput.g, throughput.b)) + 0.001f, 0.95f);
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, path_index, depth);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rand = u01(rng);
#endif // PT_RUSSIAN_ROULETTE
#endif // PT_MATERIAL_SORT

        Ray ray = pathSegments[path_index].ray;
        ShadeableIntersection isect;
        bool hit;
        if (pathSegments[path_index].remainingBounces <= 0) {
            hit = false;
        }
        else {
            hit = getClosestHit(ray, pathSegments[path_index].schannel, depth, bvhNodes, geoms, geoms_size, lightgeoms, lightgeoms_size, vertexPos, vertexNor, vertexUV, vertexSchannel, materials, isect);
        }

#if PT_MATERIAL_SORT
        if (!hit)
        {
            intersections[path_index].t = -1.0f;
            char mattype = NONE_MAT;
#if PT_LINE_RENDER
            glm::ivec2 pixel;
            glm::vec3 isectP = ray.origin + 1000000.f * ray.direction;
            if (worldToPixel(cam, isectP, pixel)) {
                int pixelIdx = pixel.x + (pixel.y * cam.resolution.x);
                float lineAlpha = lines[pixelIdx].alpha;
                //lineAlpha = glm::smoothstep(0.1f, 0.6f, lineAlpha);
                if (u01(rng) < lineAlpha) {
                    intersections[path_index].t = 1000000.f;
                    intersections[path_index].materialId = -2;
                    intersections[path_index].lightEmission = lines[pixelIdx].color;
                    mattype = LINES;
                }
            }
#endif // PT_LINE_RENDER
            pathIndices[path_index] = PACKINDEX(path_index, mattype);
        }
        else
        {
#if PT_LINE_RENDER
            // modify isect by checking line intersection
            glm::ivec2 pixel;
            glm::vec3 isectP = ray.origin + isect.t * ray.direction;
            if (worldToPixel(cam, isectP, pixel)) {
                int pixelIdx = pixel.x + (pixel.y * cam.resolution.x);
                float lineAlpha = lines[pixelIdx].alpha;
                //lineAlpha = glm::smoothstep(0.1f, 0.6f, lineAlpha);
                if (u01(rng) < lineAlpha) {
                    isect.materialId = -2 - isect.materialId;
                    isect.lightEmission = lines[pixelIdx].color;
                }
            }
#endif // PT_LINE_RENDER
            // The ray hits something
            intersections[path_index] = isect;
            // write mat type, considering rr, paths to terminate will flag their mat type with TER_
            char mattype = isect.materialId >= 0 ? mattypes[isect.materialId] : (isect.materialId == -1 ? LIGHT : LINES);
#if PT_RUSSIAN_ROULETTE
            constexpr char TER_OFFSET = DIFFUSE - TER_DIFFUSE;
            if (rand > q && mattype != LIGHT && mattype != LINES) {
                mattype += TER_OFFSET;
            }
#endif // PT_RUSSIAN_ROULETTE
            pathIndices[path_index] = PACKINDEX(path_index, mattype);
        }
#else
        if (!hit)
        {
            intersections[path_index].t = -1.0f;
#if PT_LINE_RENDER
            glm::ivec2 pixel;
            glm::vec3 isectP = ray.origin + 1000000.f * ray.direction;
            if (worldToPixel(cam, isectP, pixel)) {
                int pixelIdx = pixel.x + (pixel.y * cam.resolution.x);
                float lineAlpha = lines[pixelIdx].alpha;
                //lineAlpha = glm::smoothstep(0.1f, 0.6f, lineAlpha);
                if (u01(rng) < lineAlpha) {
                    intersections[path_index].t = 1000000.f;
                    intersections[path_index].materialId = -2;
                    intersections[path_index].lightEmission = lines[pixelIdx].color;
                }
            }
#endif // PT_LINE_RENDER
        }
        else
        {
            // The ray hits something
#if PT_LINE_RENDER
            glm::ivec2 pixel;
            glm::vec3 isectP = ray.origin + isect.t * ray.direction;
            if (worldToPixel(cam, isectP, pixel)) {
                int pixelIdx = pixel.x + (pixel.y * cam.resolution.x);
                float lineAlpha = lines[pixelIdx].alpha;
                //lineAlpha = glm::smoothstep(0.1f, 0.6f, lineAlpha);
                if (u01(rng) < lineAlpha) {
                    isect.materialId = -2 - isect.materialId;
                    isect.lightEmission = lines[pixelIdx].color;
                }
            }
            intersections[path_index] = isect;
        }
#endif // PT_LINE_RENDER
#endif // PT_MATERIAL_SORT
    }
}


__global__ void computeGBufferIntersections(
    int num_paths,
    PathSegment* pathSegments,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    glm::vec3* vertexNor,
    glm::vec4* vertexUV,
    char* vertexSchannel,
    Material* materials,
    ShadeableIntersection* intersections,
    float normalOffset = 0.f,
    PTCullingOptions culling = PTCullingOptions::CULLNONE)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        Ray ray = pathSegments[path_index].ray;
        ShadeableIntersection isect;
        bool hit = getClosestHit(ray, pathSegments[path_index].schannel, 1, bvhNodes, geoms, geoms_size, lightgeoms, lightgeoms_size, vertexPos, vertexNor, vertexUV, vertexSchannel, materials, isect, normalOffset, culling);

        if (!hit)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            intersections[path_index] = isect;
        }
    }
}

__global__ void shadeGBufferMaterial(
    int num_paths,
    PathSegment* pathSegments,
    ShadeableIntersection* shadeableIntersections,
    Material* materials,
    cudaTextureObject_t* textureHandles,
    glm::vec3* gbufferPosition,
    glm::vec3* gbufferAlbedo,
    glm::vec3* gbufferNormal)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
    PathSegment& segment = pathSegments[idx];
    ShadeableIntersection& intersection = shadeableIntersections[idx];
    int pixelIndex = segment.pixelIndex;
    glm::vec3 pos;
    glm::vec3 albedo;
    glm::vec3 nor;
    if (intersection.t > 0.0f) {
        glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;
        pos = intersectPos;
        if (intersection.materialId == -1)
        {
            albedo = 0.25f * intersection.lightEmission;
            nor = -segment.ray.direction;
        }
        else {
            glm::vec3 shadingNormal = intersection.surfaceNormal;
            Material material = materials[intersection.materialId];
            getMatParams(material, intersection, shadingNormal, textureHandles);

            albedo = 0.25f * material.color;
            nor = shadingNormal;
        }
    }
    else {
        pos = 1000000.f * segment.ray.direction + segment.ray.origin;
        albedo = glm::vec3(0.f);
        nor = glm::vec3(0.f);
    }
    gbufferPosition[pixelIndex] = pos;
    gbufferAlbedo[pixelIndex] += albedo;
    gbufferNormal[pixelIndex] = nor;
}

__global__ void shadeLineRenderMaterial(
    int num_paths,
    PathSegment* pathSegments,
    ShadeableIntersection* shadeableIntersections,
    Material* materials,
    cudaTextureObject_t* textureHandles,
    glm::vec3* gbufferPosition,
    glm::vec3* gbufferNormal,
    RenderLine* lines)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }
    PathSegment& segment = pathSegments[idx];
    ShadeableIntersection& intersection = shadeableIntersections[idx];
    int pixelIndex = segment.pixelIndex;
    RenderLine line { glm::vec3(0.f), 0.f };
    
    glm::vec3 gbufferPos = gbufferPosition[pixelIndex];
    float gbufferlDepth = glm::length((gbufferPos - segment.ray.origin));
    if (intersection.t > 0.0f && intersection.t < gbufferlDepth) {
        if (intersection.materialId >= 0) {
            glm::vec3 shadingNormal = intersection.surfaceNormal;
            Material material = materials[intersection.materialId];
            getMatParams(material, intersection, shadingNormal, textureHandles);
            if (material.linecolor.x >= 0.f) {
                line.color = material.linecolor;
                if (line.color == glm::vec3(0.f)) {
                    line.color = 0.05f * material.color;
                }
                line.alpha = 1.f;
            }
            else {
                line.alpha = 0.f;
            }
        }
    }
    if (line.alpha > 0.f) {
        lines[pixelIndex].color += 0.25f * line.color;
        lines[pixelIndex].alpha += 0.25f * line.alpha;
    }
}

#if PT_MATERIAL_SORT
__global__ void shadeMaterial(
    int iter,
    int depth, 
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    PathSegment* out_pathSegments,
    int* pathIndices,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    char* vertexSchannel,
    Material* materials,
    cudaTextureObject_t* textureHandles,
    cudaTextureObject_t envmapHandle)
#else
__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    int* bPathRemains,
    LinearBVHNode* bvhNodes,
    Geom* geoms,
    int geoms_size,
    LightGeom* lightgeoms,
    int lightgeoms_size,
    glm::vec3* vertexPos,
    char* vertexSchannel,
    Material* materials,
    cudaTextureObject_t* textureHandles,
    cudaTextureObject_t envmapHandle)
#endif // PT_MATERIAL_SORT
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_paths) {
        return;
    }

    // this is not a reference, just for reading, not writing
#if PT_MATERIAL_SORT
    int packedIndex = pathIndices[idx];
    int segmentIdx = UNPACKINDEX(packedIndex);
    int mattype = UNPACKMAT(packedIndex);
#else
    int segmentIdx = idx;
#endif // PT_MATERIAL_SORT

    PathSegment segment = pathSegments[segmentIdx];
#if PT_MATERIAL_SORT
#if PT_RUSSIAN_ROULETTE
    float q = fminf(fmaxf(segment.throughput.r, fmaxf(segment.throughput.g, segment.throughput.b)) + 0.001f, 0.95f);
#endif // PT_RUSSIAN_ROULETTE
#endif // PT_MATERIAL_SORT
    if (segment.remainingBounces > 0) {
        ShadeableIntersection intersection = shadeableIntersections[segmentIdx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
            // Set up the RNG
            // LOOK: this is how you use thrust's RNG! Please look at
            // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, segmentIdx, depth);
            thrust::uniform_real_distribution<float> u01(0, 1);

            // Hit Light
            if (intersection.materialId == -1)
            {
                float misWeight = 1.f;
#if PT_MIS
                if (depth > 0 && segment.pdf < INFINITY) {
                    misWeight = powerHeuristic(segment.pdf, intersection.pdf_Li);
                }
#endif // PT_MIS
                segment.color += misWeight * segment.throughput * intersection.lightEmission;
                segment.remainingBounces = -1;
            }
#if PT_LINE_RENDER
            // Hit Lines
            if (intersection.materialId <= -2)
            {
                segment.color += segment.throughput * intersection.lightEmission;
                segment.remainingBounces = -1;
            }
#endif // PT_LINE_RENDER
            // Otherwise, Hit Geom.
            else
            {
                glm::vec3 shadingNormal = intersection.surfaceNormal;
                glm::vec3 geoNormal = shadingNormal;
                Material material = materials[intersection.materialId];
                float lod = estimateLodByDistance(intersection.t);
                getMatParams(material, intersection, shadingNormal, textureHandles, lod);
                glm::vec3 intersectPos = intersection.t * segment.ray.direction + segment.ray.origin;

                // add material emission - not importance sampled 
                segment.color += segment.throughput * material.emission;
#if PT_MIS
                directLight(intersection.schannel, bvhNodes, geoms, geoms_size, lightgeoms, lightgeoms_size, vertexPos, vertexSchannel, segment, intersectPos, shadingNormal, geoNormal, material, rng);
#endif // PT_MIS
                Sample_f(segment, intersectPos, shadingNormal, geoNormal, material, rng);
                segment.schannel = intersection.schannel;
                segment.remainingBounces--;

#if PT_RUSSIAN_ROULETTE
#if !PT_MATERIAL_SORT
                float q = fminf(fmaxf(segment.throughput.r, fmaxf(segment.throughput.g, segment.throughput.b)) + 0.001f, 0.95f);
                float rand = u01(rng);
                if (rand > q) {
                    segment.remainingBounces = 0;
                }
#else
                if (mattype < TER_DIFFUSE)
#endif // !PT_MATERIAL_SORT
                {
                    segment.throughput /= q;
                }
#endif // PT_RUSSIAN_ROULETTE
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            // not hit
            glm::vec3 envCol = envmapHandle > 0 ? Evaluate_EnvMap(segment.ray, envmapHandle) : glm::vec3(0.f);
            segment.color += envCol * segment.throughput;
            // clear throughput
            segment.throughput = glm::vec3(0.0f);
            segment.remainingBounces = -1;
        }
    }

#if PT_MATERIAL_SORT
    out_pathSegments[idx] = segment;
#else
    bPathRemains[idx] = segment.remainingBounces > 0;
    pathSegments[idx] = segment;
#endif // PT_MATERIAL_SORT
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, int iter)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iter == 0) {
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
        else {
            float alpha = 1.f / (float)iter;
            image[iterationPath.pixelIndex] = glm::mix(image[iterationPath.pixelIndex], iterationPath.color, alpha);
        }
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int maxiter, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    constexpr dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    constexpr int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////
    // Start Tracing
    // Generate rays first
    generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths_A);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths_A + pixelcount;
    int num_paths = dev_path_end - dev_paths_A;
    int last_num_paths = num_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks
    PathSegment* dev_paths, * dev_paths_next;
    bool iterationComplete = false;
    while (!iterationComplete)
    {
        dev_paths = (depth & 0x1) ? dev_paths_B : dev_paths_A;
        dev_paths_next = (depth & 0x1) ? dev_paths_A : dev_paths_B;

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
#if PT_MATERIAL_SORT
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            cam,
            iter,
            depth,
            num_paths,
            dev_paths,
            dev_bvhnodes,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lightgeoms,
            hst_scene->lightgeoms.size(),
            dev_vertPos,
            dev_vertNor,
            dev_vertUV,
            dev_vertSchannel,
            dev_materials,
            dev_lines,
            dev_mattypes,
            dev_intersections,
            dev_pathindices
            );
#else
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            cam,
            depth,
            num_paths,
            dev_paths,
            dev_bvhnodes,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lightgeoms,
            hst_scene->lightgeoms.size(),
            dev_vertPos,
            dev_vertNor,
            dev_vertUV,
            dev_vertSchannel,
            dev_materials,
            dev_lines,
            dev_intersections
            );
#endif // PT_MATERIAL_SORT
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
#if PT_MATERIAL_SORT
        // Mat sort and compaction at one pass
        // Save mem load/store times
        int remain_num_paths = StreamCompaction::EfficientSharedMem::radixSortMatTypeCUB(num_paths, dev_pathindices, dev_pathindices);
        // copy back
        if (last_num_paths > num_paths) {
            cudaMemcpyAsync(dev_paths_next + num_paths, dev_paths + num_paths, (last_num_paths - num_paths) * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        }

        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_paths_next,
            dev_pathindices,
            dev_bvhnodes,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lightgeoms,
            hst_scene->lightgeoms.size(),
            dev_vertPos,
            dev_vertSchannel,
            dev_materials,
            dev_texurehandles,
            envmaphandle
            );
#else
        shadeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_pathremains,
            dev_bvhnodes,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lightgeoms,
            hst_scene->lightgeoms.size(),
            dev_vertPos,
            dev_vertSchannel,
            dev_materials,
            dev_texurehandles,
            envmaphandle
            );
        cudaDeviceSynchronize();

        // copy back
        if (last_num_paths > num_paths) {
            cudaMemcpyAsync(dev_paths_next + num_paths, dev_paths + num_paths, (last_num_paths - num_paths) * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        }
        // Stream Compaction
        int remain_num_paths = StreamCompaction::EfficientSharedMem::partitionStable(num_paths, sizeof(PathSegment), dev_paths_next, dev_paths, dev_pathremains);
#endif // PT_MATERIAL_SORT
        cudaDeviceSynchronize();
        checkCUDAError("shadeMaterial");

        last_num_paths = num_paths;
        num_paths = remain_num_paths;

        depth++;
        iterationComplete = (depth == traceDepth || remain_num_paths == 0);

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
#if PT_MATERIAL_SORT
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths_next, iter);
#else
    finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths, iter);
#endif // PT_MATERIAL_SORT
    ///////////////////////////////////////////////////////////////////////////

#if PT_DENOISE
#if PT_REALTIME_DENOISE
    if (iter == maxiter) {
        oidnSetFilterInt(oidn_filter, "quality", OIDN_QUALITY_HIGH);
        oidnCommitFilter(oidn_filter);
    }
    oidnExecuteFilter(oidn_filter);
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoiseimage, dev_postimage, hst_scene->postparams);
#else 
    if (iter == maxiter) {
        oidnExecuteFilter(oidn_filter);
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoiseimage, dev_postimage, hst_scene->postparams);
    }
    else {
        sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, dev_postimage, hst_scene->postparams);
        //sendImageToPBOVec4 << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, (glm::vec4*)dev_lines, dev_postimage, hst_scene->postparams);
    }
#endif // PT_REALTIME_DENOISE
#else
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image, dev_postimage, hst_scene->postparams);
#endif // PT_DENOISE

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_postimage, // dev_postimage
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void pathtraceGetGBuffer()
{
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    constexpr dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    constexpr int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////
   // Start Tracing
    for (int i = 0; i < 4; ++i) {
        glm::vec2 offset = glm::vec2((i%2), (i/2)) * 0.5f - glm::vec2(0.25f);
        // Generate rays first
        generateGBufferRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, dev_paths_A, offset);
        checkCUDAError("generate camera ray");

        int depth = 0;
        PathSegment* dev_path_end = dev_paths_A + pixelcount;
        int num_paths = dev_path_end - dev_paths_A;

        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // trace one
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

        computeGBufferIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths,
            dev_paths_A,
            dev_bvhnodes,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_lightgeoms,
            hst_scene->lightgeoms.size(),
            dev_vertPos,
            dev_vertNor,
            dev_vertUV,
            dev_vertSchannel,
            dev_materials,
            dev_intersections);
        cudaDeviceSynchronize();
        shadeGBufferMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
            num_paths,
            dev_paths_A,
            dev_intersections,
            dev_materials,
            dev_texurehandles,
            dev_GB_position,
            dev_GB_albedo,
            dev_GB_normal);

        checkCUDAError("gbuffer");

        // Line Render Preparation
#if PT_LINE_RENDER
        if (hst_scene->extend_bvhAccel->totalNodes > 0) {
            constexpr float lineRadius = (float)PT_LINE_SCALE;
            generateGBufferRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, dev_paths_A, offset);
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
            computeGBufferIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths,
                dev_paths_A,
                dev_extendbvhnodes,
                dev_geoms,
                hst_scene->geoms.size(),
                dev_lightgeoms,
                hst_scene->lightgeoms.size(),
                dev_vertPos,
                dev_vertNor,
                dev_vertUV,
                dev_vertSchannel,
                dev_materials,
                dev_intersections,
                lineRadius,
                PTCullingOptions::CULLFRONT);
            cudaDeviceSynchronize();
            shadeLineRenderMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                num_paths,
                dev_paths_A,
                dev_intersections,
                dev_materials,
                dev_texurehandles,
                dev_GB_position,
                dev_GB_normal,
                dev_lines);
        }
#endif // PT_LINE_RENDER
    }
}