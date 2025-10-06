# CUDA Path Tracer

![](saved_imgs/Sea_of_Flowers_4k.png)  
_The Sea of Flowers in Memory_

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

-   Yu Jiang
    -   [LinkedIn](https://www.linkedin.com/in/yu-jiang-450815328/)
-   Tested on: Windows 11, Ultra 7 155H @ 3.80 GHz, 32GB RAM, RTX 4060 8192MB (Personal Laptop)

## Summary

-   [Final Scene Project Overview](#final-scene-project-overview)
-   [Visual Features](#visual-features)
    -   [Anti-Aliasing](#anti-aliasing)
    -   [Perfect Specular & Refractive Material](#perfect-specular--refractive-material)
    -   [Disney BSDF](#disney-bsdf)
    -   [Texture & Normal Mapping with Arbitrary Mesh Loading](#texture-mapping--normal-mapping)
    -   [Multiple Importance Sampling](#multiple-importance-sampling)
    -   [Physically based Depth of Field](#depth-of-field)
    -   [Post-Processing](#post-processing)
    -   [Environment Mapping](#environmen-mapping)
    -   [Open Image AI Denoiser](#open-image-ai-denoiser)
    -   [Stylized Rendering](#stylized-rendering---cel-shading)
        -   [Cel-Shading](#stylized-rendering---cel-shading)
        -   [Shadow Channel](#stylized-rendering---shadow-channel)
        -   [Line Rendering](#stylized-rendering---line-rendering)
-   [Performance Features](#features)
    -   [Ray Compaction](#ray-compaction)
    -   [Material Sort](#material-sort)
    -   [Ruaaian Roulette](#russian-roulette)
    -   [Bounding Volume Hierarchies with SAH Optimization](#bounding-volume-hierarchies)
-   [Compilation Changes](#compilation-changes)
-   [Bloopers](#bloopers)
-   [References](#references)

## Final Scene Project Overview

Tell where each part of the scene comes from, show the blender scene file, show the gbuffer

## Visual Features

### Anti-Aliasing

#### Implementation

If the AA is enabled, jitter the ray on the screen within the (-0.5, 0.5) range, instead of always at the center of the pixel. So when samples increase, there will be rays shooting to different places within one pixel, so we will get soft edges.

#### Results

Here are the results of using anti-aliasing and not useing it, I also include two scaled images to better show the difference.

|                     With AA                     |                      W/O AA                       |
| :---------------------------------------------: | :-----------------------------------------------: |
|    ![](saved_imgs/cornell_UEmatball_AA.png)     |    ![](saved_imgs/cornell_UEmatball_noAA.png)     |
| ![](saved_imgs/cornell_UEmatball_AA_detail.png) | ![](saved_imgs/cornell_UEmatball_noAA_detail.png) |

### Perfect Specular & Refractive Material

#### Implementation

Implementation

#### Results

Results

### Disney BSDF

#### Implementation

Implementation

#### Results

Results

### Texture Mapping & Normal Mapping

#### Implementation

Implementation

#### Results

Results

### Multiple Importance Sampling

#### Implementation

Implementation

#### Results

Results

### Depth of Field

#### Implementation

Implementation

#### Results

Results

### Post Processing

#### Implementation

Implementation

#### Results

Results

### Environmen Mapping

#### Implementation

Implementation

#### Results

Results

### Open Image AI Denoiser

#### Implementation

Implementation

#### Results

Results

### Stylized Rendering - Cel-Shading

#### Implementation

Implementation

#### Results

Results

### Stylized Rendering - Shadow Channel

#### Implementation

Implementation

#### Results

Results

### Stylized Rendering - Line Rendering

#### Implementation

Implementation

#### Results

Results

## Performance Features

### Ray Compaction

#### Implementation

For the ray compaction, I use my compaction implementation. I use that fuction from hw02 to create a new function called partitionStable, which is very similar to radix sort but only the last digit. I saved whether this ray is going to terminate in another buffer, and use it as key to compaction to get the index, and then use another kernel to write segments into their new places.

#### Performance

We test the performance in two scenes, Open Scene with a sphere and a environment map, Closed Scene with a cornell box. We can see the performance gain of compaction here:

![](img/Compaction_time_comparison.png)

We can see that in the open scene, compaction gives much more performance gain (almost 100%), why is that? Let's think about how compaction handle the segments: it only removes the **terminated** ray segments. That means the more rays are terminated, the more performance gain it can give us, vice versa. So let's look at how many ray segments are still alive after bounces in each scene, and we will know the answer.

![](img/Num_Segments_Bounces.png)

In the open scene, number of segments decreases quickly, and that gives compaction huge space to accelerate the process by removing the terminated rays and launching less threads next time.

### Material Sort

#### Implementation

For the material sort implementation, I used `cub::DeviceRadixSort::SortKeys` (thrust also use this for radix sort), because I found my implementation cost too much time on transfering data between buffers. In my implementation, if material sort is enabled, then I will write a "mattype" key into a buffer and use it to sort. Additionally, if material sort is enabled I will predicate whether the ray is going to be terminate earliear (when hit nothing or light) and move them to the back at this stage before shading, so the compaction is automatically completed without more memory load/store cost.

#### Performance

We used three different scene to test the performance of material sort. All of the three have at least 2 different types of material (otherwise it will be obviously useless). The Basic Cornell scene is a cornell box with a sphere primitive which has 61 primitives in total. The Cornell with Blender Sphere is a larger scene with replace the sphere with blender UV sphere so it has 970 primitives in total. And the last scene we use cornell box with UE material ball we have 58600 primitives intotal which is the largest scene of the three.

![](img/MatSort_time_comparison.png)

We can see that material sort will give performance gain in small scenes, where shading materials' time takes up quite a lot in the total time, the time it spend on shading covers the cost of sorting. But for the large scene, intersect ray with triangles is the most costly part, and this part will not be boosted by material sort at all, so there will be negative impact on performance.

### Russian Roulette

#### Implementation

I judge the probability of each ray segment by the biggest number of 3 channels of their current `thoughput`, and that reflects how much the future trace is able to contribute to the final color. So the final probabilty of my implementation is `min(max(throughput.r, max(throughput.g, throughput.b)) + 0.001f, 0.95f)`. after getting the probability `q`, generate a random number and compare to this `q`, if it's lower than we keep they ray and devide the `thoughput` by `q`, otherwise we terminate this ray.

Commonly russian roulette is put after the shading part. But according to my material sort implementation I'm checking whether rays are to terminate earlier, so if that's enabled, the russian roulette probability `q` will be judged before shading in the intersection function, and tag rays to terminate with `NONE_MAT` mattype to move them to the back. And to preserve energy, in the color devision we have to use `q` judged by the color before shading, not after it.

#### Performance

Here we can see the performance gain of russian roulette in different types of scene. For the open scene we choose UE material ball with a environment map which has 58552 triangles. And for the closed scene we choose UE material ball in cornell box which has 58600 triangles.

![](img/RR_time_comparison.png)

We can figure out that russian roulette improves performance in all scenes, but it can give more performance gain in closed scene, because without russian roulette, these rays will never hit "nothing" and will always tracing in the scene, which cost a lot of performance.

### Bounding Volume Hierarchies

#### Implementation

For the BVH implementation, I look at PBRT's chapter 7.3.

![A image of BVH from PBRT](img/BVH_img1.png)

Firstly collect all the primitives in the scene, then compute the AABB of them, and then use a recursive progress on CPU to devide the AABB on the maximum extent to 2 children and work on those 2 children. In my implementation, I always use max primitives num in one leaf node is 1, since I don't want to do a more complex indexing when get intersections (But Note that this will not always be better, I just do so for convenience and time limit).

![Also an image of BVH from PBRT](img/BVH_img2_SAH.png)

For the partition method, there are 3, equal-count, equal-size, and surface area heuristic. Some times equal size/count works well like a) in the picture, but sometimes they don't, like b). So we have some better method to evaluate a _intersection cost_ after each element in the node and choose the better partition, that's the SAH(surface area heuristic). In real implementation, to lower the building time, we cannot checking the cost of partition after each primitive, that's too slow, so instead we make 10-12 _buckets_, and put primitives sequentially into buckets based on their x/y/z(based on the axis to divide with)axis value, then we only check the partition cost after each bucket.

Since BVH is a tree-like structure, it's not memory-contigent to do intersection directly on GPU directly, and the pointers to children nodes will not work after transfering data from CPU to GPU, so we have to flatten the BVH into a 1D array. In the Linear BVH node structure, we will only record the second child's index, and the first child is always just behind the parent.

To traverse BVH node in GPU, we have to use an iterative method instead of a recursive method, so we have to use a stack to record nodes index when we go deeper, so we can get back after traversing all children and pop up them. To tag the tree is fully traversed, we push a -1 at the bottom of the stack, so if we pop something and that's -1, we are end with the traversal.

#### Performance

BVH is really really really important for path tracer if you want to render scenes with lots of triangles. Why? Look at the huge difference between with and w/o BVH. We use a cornell box and a geometry exported by Blender, and then use Blender's catmull-clark subdivision to increase the number of triangles:

![](img/BVH_time_comparison3.png)

That even makes the chart hard to read! So we have to put the data into 2 charts so you can see the number.

|             With BVH              |              W/O BVH              |
| :-------------------------------: | :-------------------------------: |
| ![](img/BVH_time_comparison1.png) | ![](img/BVH_time_comparison2.png) |

VERY crazy performance increase, when the num of triangles increases exponentially, frame time w/o BVH also increases exponentially, but with BVH the frame time just goes linerly. That matches our observation about the structure: the length of the array is O(N), while the depth of the tree is only O(logN).

#### Possible Future Work

Actually, in this implementation I only implemented one level BVH, while most renders have two level BVH, BLAS and TLAS. I tried to implement them but time does not permit. How much performance that two level structures will give is hard to know, because we cannot know the gain of memory contingency and the cost of tree strcuture degeneration, which one is larger. But this can REALLY save your GPU memory, since all same mesh instances in the scene will go to the same BLAS so you don't need to put all triangles really into the scene's vertex buffer, while in my implementation I have to instanciate all meshs with their transforms and really put them into the scene. This cost about 1.343GB only for BVH for a 12.8M triangles scene, which means if the scene goes larger, then it's not able to render on a RTX 4060 with 8GB memory😭.

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
