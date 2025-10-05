# CUDA Path Tracer

![](saved_imgs/Sea_of_Flowers_4k.png)  
_The Sea of Flowers in Memory_

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

-   Yu Jiang
    -   [LinkedIn](https://www.linkedin.com/in/yu-jiang-450815328/)
-   Tested on: Windows 11, Ultra 7 155H @ 3.80 GHz, 32GB RAM, RTX 4060 8192MB (Personal Laptop)

## Summary

-   [Final Scene Project Overview](#final-scene-project-overview)
-   [Features](#features)
    -   [Ray Compaction](#ray-compaction)
    -   [Material Sort](#material-sort)
    -   [Ruaaian Roulette](#russian-roulette)
    -   [Anti-Aliasing](#anti-aliasing)
    -   [Perfect Specular & Refractive Material](#perfect-specular--refractive-material)
    -   [Disney BSDF](#disney-bsdf)
    -   [Texture & Normal Mapping with Arbitrary Mesh Loading](#texture-mapping--normal-mapping)
    -   [Multiple Importance Sampling](#multiple-importance-sampling)
    -   [Physically based Depth of Field](#depth-of-field)
    -   [Post-Processing](#post-processing)
    -   [Environment Mapping](#environmen-mapping)
    -   [Bounding Volume Hierarchies with SAH Optimization](#bounding-volume-hierarchies)
    -   [Open Image AI Denoiser](#open-image-ai-denoiser)
    -   [Stylized Rendering](#stylized-rendering---cel-shading)
        -   [Cel-Shading](#stylized-rendering---cel-shading)
        -   [Shadow Channel](#stylized-rendering---shadow-channel)
        -   [Line Rendering](#stylized-rendering---line-rendering)
-   [Compilation Changes](#compilation-changes)
-   [Bloopers](#bloopers)
-   [References](#references)

## Final Scene Project Overview

Tell where each part of the scene comes from, show the blender scene file, show the gbuffer

## Features

### Ray Compaction

#### Implementation

Implementation

#### Performance

Performance

### Material Sort

Implementation + Performance

### Russian Roulette

Implementation + Performance (Closed + Open)

### Anti-Aliasing

Implementation + Images comparison

### Perfect Specular & Refractive Material

Implementation + Images comparison

### Disney BSDF

Implementation + Images comparison

### Texture Mapping & Normal Mapping

Implementation + Images comparison

### Multiple Importance Sampling

Implementation + Performance + Images comparison

### Depth of Field

Implementation + Images comparison

### Post Processing

Implementation + Images comparison

### Environmen Mapping

Implementation + Images comparison

### Bounding Volume Hierarchies

Implementation + Performance

### Open Image AI Denoiser

Implementation + Performance + Images comparison

### Stylized Rendering - Cel-Shading

Implementation + Images comparison

### Stylized Rendering - Shadow Channel

Implementation + Images comparison

### Stylized Rendering - Line Rendering

Implementation + Images comparison

## Compilation Changes

-   Add OIDN and tiny-glTF libraries.
    -   Add headers to ./external/include, static libraries to ./external/lib, and dynamic libraries to ./external/bin
-   **NOTE: CMake file edited for libraries and new src files.**

## Bloopers

Here are some very common error I think many of us have encountered:
![](saved_imgs/blooper_epsilon2.png)  
This one is is because the default ray-sphere intersection function suffers from floating point error, So the rays generated are always hitting the same surface.

To validate the reason, I have another image like this:  
![](saved_imgs/blooper_epsilon.png)  
I have set the ior of the glass ball to 1.0 in this image, so idealy the ray will go through it without any changes, but you can see some stripes on the ball due to floating point error.

Another interesting blooper I encountered:  
![](saved_imgs/blooper_barynormal.png)  
A "sharp" sphere. That's because the barycentric coordinates is computed wrong (Yes, I don't use the glm's intersection function and it truely gives me lots of questions), so the normals computed using it is also wrong.

## References

### Tutorials

-   Bounding Volume Hierarchies, https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
-   Disney BSDF, https://schuttejoe.github.io/post/disneybsdf/
-   CUDA Textures, https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TEXTURE__OBJECT.html
-   ACES Curve, https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
-   Back-facing Line Render, https://x-wflo.github.io/2021/08/06/Cel-shading3/
-   Blender glTF 2.0 export, https://docs.blender.org/manual/en/latest/addons/import_export/scene_gltf2.html

### Libraries

-   tinygltf, https://github.com/syoyo/tinygltf
-   Intel Open Image Denoise, https://github.com/RenderKit/oidn

### Models

Please note that following websites are not accessible outside China, so you may need a VPN to read them.

-   [HSR] Castorice, official model by miHoYo. https://www.aplaybox.com/details/model/Eb6vXivegiZM
-   The Sea of Flowers in Memory, created by Wis. https://www.aplaybox.com/details/model/TBXa17wZuBZY
