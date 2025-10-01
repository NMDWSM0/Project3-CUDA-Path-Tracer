#include "sceneStructs.h"
#include <device_launch_parameters.h>
#include <stb_image.h>
#include <iostream>

static inline void CHECK(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); std::exit(1); }
}

__forceinline__ __device__ float3 decodeNormalU8(uchar4 c) {
    float3 n = make_float3(
        c.x / 255.0f * 2.0f - 1.0f,
        c.y / 255.0f * 2.0f - 1.0f,
        c.z / 255.0f * 2.0f - 1.0f);
    float len = fmaxf(1e-8f, sqrtf(n.x * n.x + n.y * n.y + n.z * n.z));
    return make_float3(n.x / len, n.y / len, n.z / len);
}

__forceinline__ __device__ uchar4 encodeNormalU8(float3 n) {
    float3 c = make_float3(0.5f * (n.x + 1.f), 0.5f * (n.y + 1.f), 0.5f * (n.z + 1.f));
    return make_uchar4(
        (unsigned char)(fminf(fmaxf(c.x, 0.f), 1.f) * 255.f + 0.5f),
        (unsigned char)(fminf(fmaxf(c.y, 0.f), 1.f) * 255.f + 0.5f),
        (unsigned char)(fminf(fmaxf(c.z, 0.f), 1.f) * 255.f + 0.5f),
        255
    );
}

__forceinline__ __device__ float3 decodeColorU8(uchar4 uc) {
    float3 c = make_float3(
        uc.x / 255.0f,
        uc.y / 255.0f,
        uc.z / 255.0f);
    return c;
}

__forceinline__ __device__ uchar4 encodeColorU8(float3 c) {
    return make_uchar4(
        (unsigned char)(fminf(fmaxf(c.x, 0.f), 1.f) * 255.f + 0.5f),
        (unsigned char)(fminf(fmaxf(c.y, 0.f), 1.f) * 255.f + 0.5f),
        (unsigned char)(fminf(fmaxf(c.z, 0.f), 1.f) * 255.f + 0.5f),
        255
    );
}

__global__ void downsampleNormalKernel(const uchar4* src, int srcW, int srcH, uchar4* dst, int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    int sx = x * 2;
    int sy = y * 2;

    uchar4 c00 = src[glm::min(sy, srcH - 1) * srcW + glm::min(sx, srcW - 1)];
    uchar4 c10 = src[glm::min(sy, srcH - 1) * srcW + glm::min(sx + 1, srcW - 1)];
    uchar4 c01 = src[glm::min(sy + 1, srcH - 1) * srcW + glm::min(sx, srcW - 1)];
    uchar4 c11 = src[glm::min(sy + 1, srcH - 1) * srcW + glm::min(sx + 1, srcW - 1)];

    float3 n00 = decodeNormalU8(c00);
    float3 n10 = decodeNormalU8(c10);
    float3 n01 = decodeNormalU8(c01);
    float3 n11 = decodeNormalU8(c11);

    float3 n = make_float3(
        n00.x + n10.x + n01.x + n11.x,
        n00.y + n10.y + n01.y + n11.y,
        n00.z + n10.z + n01.z + n11.z);

    float len = fmaxf(1e-8f, sqrtf(n.x * n.x + n.y * n.y + n.z * n.z));
    n.x /= len; n.y /= len; n.z /= len;

    dst[y * dstW + x] = encodeNormalU8(n);
}

__global__ void downsampleColorKernel(const uchar4* src, int srcW, int srcH, uchar4* dst, int dstW, int dstH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    int sx = x * 2;
    int sy = y * 2;

    uchar4 uc00 = src[glm::min(sy, srcH - 1) * srcW + glm::min(sx, srcW - 1)];
    uchar4 uc10 = src[glm::min(sy, srcH - 1) * srcW + glm::min(sx + 1, srcW - 1)];
    uchar4 uc01 = src[glm::min(sy + 1, srcH - 1) * srcW + glm::min(sx, srcW - 1)];
    uchar4 uc11 = src[glm::min(sy + 1, srcH - 1) * srcW + glm::min(sx + 1, srcW - 1)];

    float3 c00 = decodeColorU8(uc00);
    float3 c10 = decodeColorU8(uc10);
    float3 c01 = decodeColorU8(uc01);
    float3 c11 = decodeColorU8(uc11);

    float3 c = make_float3(
        (c00.x + c10.x + c01.x + c11.x) * .25f,
        (c00.y + c10.y + c01.y + c11.y) * .25f,
        (c00.z + c10.z + c01.z + c11.z) * .25f);

    dst[y * dstW + x] = encodeColorU8(c);
}

int calcMipLevels(int w, int h) {
    int m = std::max(w, h);
    int lv = 1;
    while (m > 1) { m >>= 1; ++lv; }
    return lv;
}

void Texture::loadToCPU(const std::string& filename)
{
    size_t dotpos = filename.find_last_of('.');
    isHDR = false;
    if (dotpos != std::string::npos) {
        if (filename[dotpos + 1] == 'h' && filename[dotpos + 2] == 'd' && filename[dotpos + 3] == 'r') {
            isHDR = true;
        }
    }

    if (isHDR) {
        float* pixels = stbi_loadf(filename.c_str(), &w, &h, &c, 4);
        if (!pixels) {
            printf("failed to load texture: %s\n", filename.c_str());
            std::exit(1);
        }
        // load to CPU first
        cpudataHDR.resize(w * h * 4 * sizeof(float));
        memcpy(cpudataHDR.data(), pixels, w * h * 4 * sizeof(float));
    }
    else {
        stbi_uc* pixels = stbi_load(filename.c_str(), &w, &h, &c, 4);
        if (!pixels) {
            printf("failed to load texture: %s\n", filename.c_str());
            std::exit(1);
        }
        // load to CPU first
        cpudata.resize(w * h * 4 * sizeof(unsigned char));
        memcpy(cpudata.data(), pixels, w * h * 4 * sizeof(unsigned char));
    }
    c = 4;
}

void Texture::loadToCPU(unsigned char* data, int w, int h, int c) 
{
    this->w = w; this->h = h; this->c = c;
    // should be 4-channel uchar since we are using customed gltf image loading function
    cpudata.resize(w * h * 4 * sizeof(unsigned char));
    memcpy(cpudata.data(), data, w * h * 4 * sizeof(unsigned char));
    c = 4;
}

cudaTextureObject_t Texture::loadToCuda() 
{
    if ((isHDR && cpudataHDR.size() == 0) || (!isHDR && cpudata.size() == 0)) {
        return 0;
    }
    levels = isHDR ? 1 : calcMipLevels(w, h);

    cudaChannelFormatKind format = isHDR ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    cudaChannelFormatDesc ch = isHDR ? cudaCreateChannelDesc<float4>() : cudaCreateChannelDesc<uchar4>();
    cudaExtent extent{ (size_t)w, (size_t)h, 0 };
    CHECK(cudaMallocMipmappedArray(&array, &ch, extent, levels), "cudaMallocMipmappedArray");

    // copy level 0
    cudaArray_t level0;
    cudaGetMipmappedArrayLevel(&level0, array, 0);
    if (isHDR) {
        CHECK(
            cudaMemcpyToArray(level0, 0, 0, cpudataHDR.data(), w * h * sizeof(float4), cudaMemcpyHostToDevice),
            "cudaMemcpyToArrayHDR");
    }
    else {
        CHECK(
            cudaMemcpyToArray(level0, 0, 0, cpudata.data(), w * h * sizeof(uchar4), cudaMemcpyHostToDevice),
            "cudaMemcpyToArray");
    }

    // generate mipmap
    if (!isHDR) {
        uchar4* dev_src;
        uchar4* dev_dst;
        cudaMalloc(&dev_src, w * h * sizeof(uchar4));
        cudaMalloc(&dev_dst, w * h * sizeof(uchar4));

        auto downsampleNormalLevel = [&](int srcLevel, int dstLevel, int srcW, int srcH, int dstW, int dstH)
            {
                cudaArray_t srcArr, dstArr;
                cudaGetMipmappedArrayLevel(&srcArr, array, srcLevel);
                cudaGetMipmappedArrayLevel(&dstArr, array, dstLevel);

                cudaMemcpyFromArray(dev_src, srcArr, 0, 0, srcW * srcH * sizeof(uchar4), cudaMemcpyDeviceToDevice);

                dim3 block(16, 16);
                dim3 grid((dstW + 15) / 16, (dstH + 15) / 16);
                if (isNormal) {
                    downsampleNormalKernel << <grid, block >> > (dev_src, srcW, srcH, dev_dst, dstW, dstH);
                }
                else {
                    downsampleColorKernel << <grid, block >> > (dev_src, srcW, srcH, dev_dst, dstW, dstH);
                }
                cudaDeviceSynchronize();

                cudaMemcpyToArray(dstArr, 0, 0, dev_dst, dstW * dstH * sizeof(uchar4), cudaMemcpyDeviceToDevice);
            };

        int _w = w, _h = h;
        for (int lv = 1; lv < levels; ++lv) {
            int srcW = std::max(1, _w);
            int srcH = std::max(1, _h);
            _w = std::max(1, _w / 2);
            _h = std::max(1, _h / 2);
            downsampleNormalLevel(lv - 1, lv, srcW, srcH, _w, _h);
        }
        
        cudaFree(dev_src);
        cudaFree(dev_dst);
    }

    cudaResourceDesc res{};
    res.resType = cudaResourceTypeMipmappedArray;
    res.res.mipmap.mipmap = array;
    res.res.linear.desc = ch;
    res.res.linear.sizeInBytes = w * h * (isHDR ? sizeof(float4) : sizeof(uchar4));

    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.mipmapFilterMode = cudaFilterModeLinear;
    tex.minMipmapLevelClamp = 0.0f;
    tex.maxMipmapLevelClamp = float(levels - 1);
    tex.readMode = isHDR ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    tex.normalizedCoords = 1; 

    CHECK(cudaCreateTextureObject(&handle, &res, &tex, nullptr), "cudaCreateTextureObject");
    return handle;
}

void Texture::FreeCudaSide()
{
    cudaDestroyTextureObject(handle);
    cudaFreeMipmappedArray(array);

    handle = 0;
    array = nullptr;
}

Texture::~Texture()
{
    if (array) {
        FreeCudaSide();
    }
    cpudata.swap(std::vector<unsigned char>());
    cpudataHDR.swap(std::vector<float>());
}