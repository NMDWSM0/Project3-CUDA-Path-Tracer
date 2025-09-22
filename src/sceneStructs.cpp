#include "sceneStructs.h"

#include "stb_image.h"
#include <iostream>

static inline void CHECK(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); std::exit(1); }
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
}

cudaTextureObject_t Texture::loadToCuda() 
{
    if ((isHDR && cpudataHDR.size() == 0) || (!isHDR && cpudata.size() == 0)) {
        return 0;
    }

    cudaChannelFormatKind format = isHDR ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
    cudaChannelFormatDesc ch = isHDR ? cudaCreateChannelDesc<float4>() : cudaCreateChannelDesc<uchar4>();
    CHECK(cudaMallocArray(&array, &ch, w, h), "cudaMallocArray");

    if (isHDR) {
        CHECK(
            cudaMemcpyToArray(array, 0, 0, cpudataHDR.data(), w * h * sizeof(float4), cudaMemcpyHostToDevice),
            "cudaMemcpyToArrayHDR");
    }
    else {
        CHECK(
            cudaMemcpyToArray(array, 0, 0, cpudata.data(), w * h * sizeof(uchar4), cudaMemcpyHostToDevice),
            "cudaMemcpyToArray");
    }

    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = array;
    res.res.linear.desc = ch;
    res.res.linear.sizeInBytes = w * h * (isHDR ? sizeof(float4) : sizeof(uchar4));

    cudaTextureDesc tex{};
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;
    tex.filterMode = cudaFilterModeLinear;
    tex.readMode = isHDR ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
    tex.normalizedCoords = 1; 

    CHECK(cudaCreateTextureObject(&handle, &res, &tex, nullptr), "cudaCreateTextureObject");
    return handle;
}

void Texture::FreeCudaSide()
{
    cudaDestroyTextureObject(handle);
    cudaFreeArray(array);

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