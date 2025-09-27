// print BVH build info
#define BVH_PRINT_BUILD_INFO 1

// error check
#define PT_ERRORCHECK 0

// material sort
#define PT_MATERIAL_SORT 1

// russian roulette
#define PT_RUSSIAN_ROULETTE 1

// multiple importance sampling
#define PT_MIS 1

// anti-aliasing
#define PT_AA 1

// depth of field
#define PT_DOF 1

// OIDN denoise
#define PT_DENOISE 1

// denoise realtimely or not
#define PT_REALTIME_DENOISE 0

// normal map G reverse
#define PT_OPENGL_NORMALMAP 0

// use BVH intersection
#define PT_USEBVH 1

// toon shading
#define PT_TOON_SHADING 1

#if PT_TOON_SHADING
#define PT_CEL_SHADING 1
#define PT_SHADOW_CHANNEL 1
#define PT_LINE_RENDER 1
#endif