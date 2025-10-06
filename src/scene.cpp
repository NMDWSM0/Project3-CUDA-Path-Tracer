#include "scene.h"
#include "defines.h"
#define TINYGLTF_IMPLEMENTATION

#include "utilities.h"
#include "postprocess.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <stb_image.h>
#include <tiny_gltf.h>

using namespace std;
using json = nlohmann::json;

struct GLTFMeshPrim {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec4> uvs;
    std::vector<char> schannels;
    std::vector<int> indices;
    int sceneMatIdx;
};

static std::map<int, std::vector<GLTFMeshPrim>> MeshPrims;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

struct MeshInstance {
    int node = -1;
    int mesh = -1;
    glm::mat4 world{ 1.0f };
};

struct CameraInstance {
    int node = -1;
    int camera = -1;
    glm::mat4 world{ 1.0f };
};

struct LightInstance {
    int node = -1;
    int light = -1;
    glm::mat4 world{ 1.0f };
};

static inline glm::mat4 LocalOf(const tinygltf::Node& n) {
    if (n.matrix.size() == 16) {
        return glm::make_mat4(reinterpret_cast<const float*>(n.matrix.data()));
    }
    glm::vec3 T(0.0f), S(1.0f);
    glm::quat R(1, 0, 0, 0); // w,x,y,z
    if (n.translation.size() == 3) 
        T = { (float)n.translation[0], (float)n.translation[1], (float)n.translation[2] };
    if (n.scale.size() == 3)      
        S = { (float)n.scale[0], (float)n.scale[1], (float)n.scale[2] };
    if (n.rotation.size() == 4)    
        R = glm::quat((float)n.rotation[3], (float)n.rotation[0], (float)n.rotation[1], (float)n.rotation[2]);
    return glm::translate(glm::mat4(1.0f), T) * glm::mat4_cast(R) * glm::scale(glm::mat4(1.0f), S); // T*R*S
}

static void DFS(
    const tinygltf::Model& m, 
    int ni, 
    const glm::mat4& parent, 
    std::vector<MeshInstance>& out, 
    std::vector<CameraInstance>& outcams,
    std::vector<LightInstance>& outlights)
{
    const auto& n = m.nodes[ni];
    glm::mat4 world = parent * LocalOf(n);
    if (n.mesh >= 0)
    {
        out.push_back({ ni, n.mesh, world });
    }
    else if (n.camera >= 0)
    {
        outcams.push_back({ ni, n.camera, world });
    }   
    else if (n.extensions.find("KHR_lights_punctual") != n.extensions.end()) {
        const auto& ext = n.extensions.at("KHR_lights_punctual");
        outlights.push_back({ ni, ext.Get("light").Get<int>(), world });
    }
        
    for (int c : n.children) 
        DFS(m, c, world, out, outcams, outlights);
}

static void CollectInstancesOneScene(
    const tinygltf::Model& model, 
    const glm::mat4& inputTransform,
    std::vector<MeshInstance>& out, 
    std::vector<CameraInstance>& outcams, 
    std::vector<LightInstance>& outlights)
{
    const int s = (model.defaultScene >= 0) ? model.defaultScene : 0;
    for (int root : model.scenes[s].nodes) 
        DFS(model, root, inputTransform, out, outcams, outlights);
}

static void loadMeshes(Scene* scene, tinygltf::Model& gltfModel)
{
    //glm::mat3 normalTransform = glm::inverseTranspose(glm::mat3(inputTransform));
    for (int gltfMeshIdx = 0; gltfMeshIdx < gltfModel.meshes.size(); gltfMeshIdx++)
    {
        tinygltf::Mesh gltfMesh = gltfModel.meshes[gltfMeshIdx];
        // add a new mesh
        // initialize prims vector (a mesh can have multiple prims)
        std::vector<GLTFMeshPrim> GLTF_prims;

        for (int gltfPrimIdx = 0; gltfPrimIdx < gltfMesh.primitives.size(); gltfPrimIdx++)
        {
            tinygltf::Primitive prim = gltfMesh.primitives[gltfPrimIdx];
            GLTF_prims.push_back(GLTFMeshPrim());
            GLTFMeshPrim& meshprim = GLTF_prims[GLTF_prims.size() - 1];

            // Skip points and lines
            if (prim.mode != TINYGLTF_MODE_TRIANGLES)
                continue;

            int indicesIndex = prim.indices;
            int positionIndex = -1;
            int normalIndex = -1;
            int uv0Index = -1;
            int uv1Index = -1;
            int schannelIndex = -1;

            if (prim.attributes.count("POSITION") > 0)
            {
                positionIndex = prim.attributes["POSITION"];
            }

            if (prim.attributes.count("NORMAL") > 0)
            {
                normalIndex = prim.attributes["NORMAL"];
            }

            if (prim.attributes.count("TEXCOORD_0") > 0)
            {
                uv0Index = prim.attributes["TEXCOORD_0"];
            }

            if (prim.attributes.count("TEXCOORD_1") > 0)
            {
                uv1Index = prim.attributes["TEXCOORD_1"];
            }

            if (prim.attributes.count("_SCHANNEL") > 0)
            {
                schannelIndex = prim.attributes["_SCHANNEL"];
            }

            // Vertex positions
            tinygltf::Accessor positionAccessor = gltfModel.accessors[positionIndex];
            tinygltf::BufferView positionBufferView = gltfModel.bufferViews[positionAccessor.bufferView];
            const tinygltf::Buffer& positionBuffer = gltfModel.buffers[positionBufferView.buffer];
            const uint8_t* positionBufferAddress = positionBuffer.data.data();
            int positionStride = tinygltf::GetComponentSizeInBytes(positionAccessor.componentType) * tinygltf::GetNumComponentsInType(positionAccessor.type);
            if (positionBufferView.byteStride > 0)
                positionStride = positionBufferView.byteStride;

            // Vertex indices
            tinygltf::Accessor indexAccessor = gltfModel.accessors[indicesIndex];
            tinygltf::BufferView indexBufferView = gltfModel.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer& indexBuffer = gltfModel.buffers[indexBufferView.buffer];
            const uint8_t* indexBufferAddress = indexBuffer.data.data();
            int indexStride = tinygltf::GetComponentSizeInBytes(indexAccessor.componentType) * tinygltf::GetNumComponentsInType(indexAccessor.type);

            // Normals
            tinygltf::Accessor normalAccessor;
            tinygltf::BufferView normalBufferView;
            const uint8_t* normalBufferAddress = nullptr;
            int normalStride = -1;
            if (normalIndex > -1)
            {
                normalAccessor = gltfModel.accessors[normalIndex];
                normalBufferView = gltfModel.bufferViews[normalAccessor.bufferView];
                const tinygltf::Buffer& normalBuffer = gltfModel.buffers[normalBufferView.buffer];
                normalBufferAddress = normalBuffer.data.data();
                normalStride = tinygltf::GetComponentSizeInBytes(normalAccessor.componentType) * tinygltf::GetNumComponentsInType(normalAccessor.type);
                if (normalBufferView.byteStride > 0)
                    normalStride = normalBufferView.byteStride;
            }

            // Texture coordinates
            tinygltf::Accessor uv0Accessor;
            tinygltf::BufferView uv0BufferView;
            const uint8_t* uv0BufferAddress = nullptr;
            int uv0Stride = -1;
            if (uv0Index > -1)
            {
                uv0Accessor = gltfModel.accessors[uv0Index];
                uv0BufferView = gltfModel.bufferViews[uv0Accessor.bufferView];
                const tinygltf::Buffer& uv0Buffer = gltfModel.buffers[uv0BufferView.buffer];
                uv0BufferAddress = uv0Buffer.data.data();
                uv0Stride = tinygltf::GetComponentSizeInBytes(uv0Accessor.componentType) * tinygltf::GetNumComponentsInType(uv0Accessor.type);
                if (uv0BufferView.byteStride > 0)
                    uv0Stride = uv0BufferView.byteStride;
            }
            tinygltf::Accessor uv1Accessor;
            tinygltf::BufferView uv1BufferView;
            const uint8_t* uv1BufferAddress = nullptr;
            int uv1Stride = -1;
            if (uv1Index > -1)
            {
                uv1Accessor = gltfModel.accessors[uv1Index];
                uv1BufferView = gltfModel.bufferViews[uv1Accessor.bufferView];
                const tinygltf::Buffer& uv1Buffer = gltfModel.buffers[uv1BufferView.buffer];
                uv1BufferAddress = uv1Buffer.data.data();
                uv1Stride = tinygltf::GetComponentSizeInBytes(uv1Accessor.componentType) * tinygltf::GetNumComponentsInType(uv1Accessor.type);
                if (uv1BufferView.byteStride > 0)
                    uv1Stride = uv1BufferView.byteStride;
            }

            // Shadow channel
            tinygltf::Accessor schannelAccessor;
            tinygltf::BufferView schannelBufferView;
            const uint8_t* schannelBufferAddress = nullptr;
            int schannelStride = -1;
            if (schannelIndex > -1)
            {
                schannelAccessor = gltfModel.accessors[schannelIndex];
                schannelBufferView = gltfModel.bufferViews[schannelAccessor.bufferView];
                const tinygltf::Buffer& schannelBuffer = gltfModel.buffers[schannelBufferView.buffer];
                schannelBufferAddress = schannelBuffer.data.data();
                schannelStride = tinygltf::GetComponentSizeInBytes(schannelAccessor.componentType) * tinygltf::GetNumComponentsInType(schannelAccessor.type);
                if (schannelBufferView.byteStride > 0)
                    schannelStride = schannelBufferView.byteStride;
            }

            std::vector<glm::vec3>& vertices = meshprim.vertices;
            std::vector<glm::vec3>& normals = meshprim.normals;
            std::vector<glm::vec4>& uvs = meshprim.uvs;
            std::vector<char>& schannels = meshprim.schannels;

            // Get vertex data
            for (size_t vertexIndex = 0; vertexIndex < positionAccessor.count; vertexIndex++)
            {
                glm::vec3 vertex, normal, tangent;
                glm::vec2 uv0, uv1;
                float schannel = 0;

                {
                    const uint8_t* address = positionBufferAddress + positionBufferView.byteOffset + positionAccessor.byteOffset + (vertexIndex * positionStride);
                    memcpy(&vertex, address, sizeof(glm::vec3));
                }

                if (normalIndex > -1)
                {
                    const uint8_t* address = normalBufferAddress + normalBufferView.byteOffset + normalAccessor.byteOffset + (vertexIndex * normalStride);
                    memcpy(&normal, address, sizeof(glm::vec3));
                }

                if (uv0Index > -1)
                {
                    const uint8_t* address = uv0BufferAddress + uv0BufferView.byteOffset + uv0Accessor.byteOffset + (vertexIndex * uv0Stride);
                    memcpy(&uv0, address, sizeof(glm::vec2));
                }
                if (uv1Index > -1)
                {
                    const uint8_t* address = uv1BufferAddress + uv1BufferView.byteOffset + uv1Accessor.byteOffset + (vertexIndex * uv1Stride);
                    memcpy(&uv1, address, sizeof(glm::vec2));
                }

                if (schannelIndex > -1)
                {
                    const uint8_t* address = schannelBufferAddress + schannelBufferView.byteOffset + schannelAccessor.byteOffset + (vertexIndex * schannelStride);
                    memcpy(&schannel, address, sizeof(float));
                }

                vertices.push_back(vertex);
                normals.push_back(normal);
                uvs.push_back(glm::vec4(uv0, uv1));
                schannels.push_back(static_cast<char>((int)schannel));
            }

            // Get index data
            std::vector<int> indices(indexAccessor.count);
            const uint8_t* baseAddress = indexBufferAddress + indexBufferView.byteOffset + indexAccessor.byteOffset;
            if (indexStride == 1)
            {
                std::vector<uint8_t> quarter;
                quarter.resize(indexAccessor.count);

                memcpy(quarter.data(), baseAddress, (indexAccessor.count * indexStride));

                // Convert quarter precision indices to full precision
                for (size_t i = 0; i < indexAccessor.count; i++)
                {
                    indices[i] = quarter[i];
                }
            }
            else if (indexStride == 2)
            {
                std::vector<uint16_t> half;
                half.resize(indexAccessor.count);

                memcpy(half.data(), baseAddress, (indexAccessor.count * indexStride));

                // Convert half precision indices to full precision
                for (size_t i = 0; i < indexAccessor.count; i++)
                {
                    indices[i] = half[i];
                }
            }
            else
            {
                memcpy(indices.data(), baseAddress, (indexAccessor.count * indexStride));
            }
            meshprim.indices = std::move(indices);
            meshprim.sceneMatIdx = prim.material + scene->materials.size();
        }

        // then push those prims
        MeshPrims[gltfMeshIdx] = std::move(GLTF_prims);
    }
}

static void loadMeshInstances(Scene* scene, const std::vector<MeshInstance>& instances, int overridematIndex) {
    for (auto& instance : instances) {
        glm::mat4 positionTransform = instance.world;
        glm::mat3 normalTransform = glm::inverseTranspose(glm::mat3(positionTransform));

        // instance the mesh to scene
        std::vector<GLTFMeshPrim>& GLTF_prims = MeshPrims[instance.mesh];
        // instance all prims in this mesh
        for (auto& prim : GLTF_prims) 
        {
			// push geometrys, be careful at mat/vert index offset
			int vertBase = scene->vertPos.size();
			int sceneMatIdx = prim.sceneMatIdx;
			for (int tid = 0; tid * 3 < prim.indices.size(); ++tid) {
				Geom newGeom(TRIANGLE);
				newGeom.vertIds = glm::ivec3(prim.indices[3 * tid], prim.indices[3 * tid + 1], prim.indices[3 * tid + 2]) + glm::ivec3(vertBase);
                if (overridematIndex >= 0) {
                    newGeom.materialid = overridematIndex;
                }
                else {
                    newGeom.materialid = sceneMatIdx;
                }
				scene->geoms.push_back(newGeom);
			}

			// push pos, nor and uv
			scene->vertPos.insert(scene->vertPos.end(), prim.vertices.begin(), prim.vertices.end());
			scene->vertNor.insert(scene->vertNor.end(), prim.normals.begin(), prim.normals.end());
			scene->vertUV.insert(scene->vertUV.end(), prim.uvs.begin(), prim.uvs.end());
			scene->vertSchannel.insert(scene->vertSchannel.end(), prim.schannels.begin(), prim.schannels.end());

            // apply transforms for positions and normals
            int last_index = scene->vertPos.size();
            int first_index = last_index - prim.vertices.size();
            for (int i = first_index; i < last_index; ++i) {
                scene->vertPos[i] = glm::vec3(positionTransform * glm::vec4(scene->vertPos[i], 1.0f));
                scene->vertNor[i] = glm::normalize(normalTransform * scene->vertNor[i]);
            }
        }
    }
}

// customed image load func, force 4 channels
static bool customLoadImageData(tinygltf::Image* image, int image_idx, std::string* err,
    std::string* warn, int req_width, int req_height,
    const unsigned char* bytes, int size, void* user_data) {
    int w, h, comp;
    unsigned char* data = stbi_load_from_memory(bytes, size, &w, &h, &comp, 4);

    if (!data) {
        if (err) *err = "Failed to load image";
        return false;
    }

    image->width = w;
    image->height = h;
    image->component = 4;
    image->bits = 8;
    image->pixel_type = TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE;
    image->image.assign(data, data + w * h * 4);
    stbi_image_free(data);

    return true;
}

void loadTextures(Scene* scene, tinygltf::Model& gltfModel, const std::vector<bool>& isNormal)
{
    for (size_t i = 0; i < gltfModel.textures.size(); ++i)
    {
        tinygltf::Texture& gltfTex = gltfModel.textures[i];
        tinygltf::Image& image = gltfModel.images[gltfTex.source];
        Texture texture;
        texture.isNormal = isNormal[i];
        texture.loadToCPU(image.image.data(), image.width, image.height, image.component);
        scene->textures.push_back(texture);
    }
}

void loadMaterials(Scene* scene, tinygltf::Model& gltfModel, std::vector<bool>& isNormal)
{
    int sceneTexIdx = scene->textures.size();
    for (size_t i = 0; i < gltfModel.materials.size(); i++)
    {
        const tinygltf::Material gltfMaterial = gltfModel.materials[i];
        const tinygltf::PbrMetallicRoughness pbr = gltfMaterial.pbrMetallicRoughness;

        // Convert glTF material
        Material material;
        material.type = DISNEY;

        // Albedo
        material.color = glm::vec3((float)pbr.baseColorFactor[0], (float)pbr.baseColorFactor[1], (float)pbr.baseColorFactor[2]);
        if (pbr.baseColorTexture.index > -1)
        {
            material.baseColorTexId = pbr.baseColorTexture.index + sceneTexIdx;
            material.baseColorTexUV = pbr.baseColorTexture.texCoord;
        }

        // Emission
        material.emission = glm::vec3((float)gltfMaterial.emissiveFactor[0], (float)gltfMaterial.emissiveFactor[1], (float)gltfMaterial.emissiveFactor[2]);
        if (gltfMaterial.emissiveTexture.index > -1)
        {
            material.emissionmapTexId = gltfMaterial.emissiveTexture.index + sceneTexIdx;
            material.emissionmapTexUV = gltfMaterial.emissiveTexture.texCoord;
        }
            
        // Roughness and Metallic
        material.roughness = (float)pbr.roughnessFactor;
        material.roughness = glm::clamp(material.roughness * material.roughness, 0.001f, 1.f);
        material.metallic = (float)pbr.metallicFactor;
        if (pbr.metallicRoughnessTexture.index > -1) 
        {
            material.metallicRoughnessTexId = pbr.metallicRoughnessTexture.index + sceneTexIdx;
            material.metallicRoughnessTexUV = pbr.metallicRoughnessTexture.texCoord;
        }
            
        // Normal Map
        if (gltfMaterial.normalTexture.index > -1 && gltfMaterial.normalTexture.scale >= 0) {
            material.normalmapTexId = gltfMaterial.normalTexture.index + sceneTexIdx;
            material.normalmapTexUV = gltfMaterial.normalTexture.texCoord;
            material.normalStrength = gltfMaterial.normalTexture.scale;
            isNormal[gltfMaterial.normalTexture.index] = true;
        }

        // KHR_materials_transmission
        material.transmission = 0.f;
        if (gltfMaterial.extensions.find("KHR_materials_transmission") != gltfMaterial.extensions.end())
        {
            const auto& ext = gltfMaterial.extensions.at("KHR_materials_transmission");
            if (ext.Has("transmissionFactor"))
                material.transmission = (float)(ext.Get("transmissionFactor").Get<double>());
            if (ext.Has("transmissionTexture")) {
                material.transmissionmapTexId = ext.Get("transmissionTexture").Get("index").Get<int>() + sceneTexIdx;
                material.transmissionmapTexUV = ext.Get("transmissionTexture").Get("texCoord").Get<int>();
            }
        }

        // KHR_materials_ior
        material.ior = 1.5f;
        if (gltfMaterial.extensions.find("KHR_materials_ior") != gltfMaterial.extensions.end())
        {
            const auto& ext = gltfMaterial.extensions.at("KHR_materials_ior");
            if (ext.Has("ior"))
                material.ior = (float)(ext.Get("ior").Get<double>());
        }

        // KHR_materials_clearcoat
        material.clearcoat = 0.f;
        material.coatroughness = 0.001f;
        if (gltfMaterial.extensions.find("KHR_materials_clearcoat") != gltfMaterial.extensions.end())
        {
            const auto& ext = gltfMaterial.extensions.at("KHR_materials_clearcoat");
            if (ext.Has("clearcoatFactor"))
                material.clearcoat = (float)(ext.Get("clearcoatFactor").Get<double>());
            if (ext.Has("clearcoatRoughnessFactor"))
                material.coatroughness = fmax((float)(ext.Get("clearcoatRoughnessFactor").Get<double>()), 0.001f);
        }

        // KHR_materials_emissive_strength
        material.emissionStrength = 1.f;
        if (gltfMaterial.extensions.find("KHR_materials_emissive_strength") != gltfMaterial.extensions.end())
        {
            const auto& ext = gltfMaterial.extensions.at("KHR_materials_emissive_strength");
            if (ext.Has("emissiveStrength"))
                material.emissionStrength = (float)(ext.Get("emissiveStrength").Get<double>());
        }

        // my_subsurface
        material.subsurface = 0.f;
        if (gltfMaterial.extras.Has("my_subsurface")) {
            material.subsurface = (float)(gltfMaterial.extras.Get("my_subsurface").Get<double>());
        }

        // line color
        // < 0 : do not draw lines (default)
        // 0.0 : draw lines, and use object color mix with black for line color
        // 0 - 1 : draw lines, and use the input RGB color
        material.linecolor = glm::vec3(-2.f);
        if (gltfMaterial.extras.Has("my_linecolor")) {
            float r = gltfMaterial.extras.Get("my_linecolor").Get(0).GetNumberAsDouble();
            float g = gltfMaterial.extras.Get("my_linecolor").Get(1).GetNumberAsDouble();
            float b = gltfMaterial.extras.Get("my_linecolor").Get(2).GetNumberAsDouble();
            material.linecolor = glm::vec3(r, g, b);
            if (r > 0.f || g > 0.f || b > 0.f) {
                material.linecolor = srgbToLinear(material.linecolor);
            }
        }

        // toon shading
        material.toonshading = false;
        if (gltfMaterial.extras.Has("my_toonshading")) {
            material.toonshading = gltfMaterial.extras.Get("my_toonshading").Get<bool>();
        }

        scene->materials.push_back(material);
    }
    // Default material
    if (scene->materials.size() == 0)
    {
        Material defaultMat;
        scene->materials.push_back(defaultMat);
    }
}

void loadCamera(Scene* scene, tinygltf::Model& gltfModel, const std::vector<CameraInstance>& caminstances) {
    if (caminstances.size() > 0) {
        const auto& caminst = caminstances[0];
        const auto& _cam = gltfModel.cameras[caminst.camera];  // load the first camera
        if (_cam.type == "perspective") {
            const auto& cam = _cam.perspective;
            const auto& transform = caminst.world;
            glm::vec3 position = glm::vec3(transform[3]);
            glm::vec3 right = glm::normalize(glm::vec3(transform[0]));
            glm::vec3 up = glm::normalize(glm::vec3(transform[1]));
            glm::vec3 forward = -glm::normalize(glm::vec3(transform[2]));
            glm::vec3 lookAt = position + forward * max(scene->state.camera.focalDistance, 1.f);
            // position and rotation
            scene->state.camera.position = position;
            scene->state.camera.lookAt = lookAt;
            scene->state.camera.up = up;
            scene->state.camera.view = forward;
            scene->state.camera.right = glm::normalize(glm::cross(scene->state.camera.view, scene->state.camera.up));
            // calculate fov based on resolution
            float fovy = cam.yfov * 180 * INV_PI;
            float yscaled = tan(cam.yfov * 0.5f);
            float xscaled = (yscaled * scene->state.camera.resolution.x) / scene->state.camera.resolution.y;
            float fovx = atan(xscaled) * 180 * INV_PI;
            scene->state.camera.fov = glm::vec2(fovx, fovy);
            scene->state.camera.pixelLength = glm::vec2(2 * xscaled / (float)scene->state.camera.resolution.x,
                2 * yscaled / (float)scene->state.camera.resolution.y);
        }
        else {
            // don't load that
        }
    }
}

void loadLights(Scene* scene, tinygltf::Model& gltfModel, const std::vector<LightInstance>& lightinstances) {
    // VERY VERY IMPORTANT: when exporting from Blender, choose "unitless" light data!!!
    for (int i = 0; i < lightinstances.size(); ++i)
    {
        const auto& lightinstance = lightinstances[i];
        const auto& _light = gltfModel.lights[lightinstance.light];  // load the first camera
        if (_light.type == "point") {
            // add a sphere light to scene
            LightGeom newLight(SPHERELIGHT);
            glm::vec3 emission = glm::vec3(_light.color[0], _light.color[1], _light.color[2]) * (float)_light.intensity;
            glm::vec3 position = glm::vec3(lightinstance.world[3]);
            newLight.emission = emission;
            newLight.position = position;
            if (_light.extras.Has("radius")) {
                newLight.radius = fmax(_light.extras.Get("radius").Get<double>(), 0.01f);
            }
            else {
                newLight.radius = 0.5f;
            }
            newLight.emission /= PI * newLight.radius * newLight.radius;
            scene->lightgeoms.push_back(newLight);
        }
        else if (_light.type == "spot") {
            // add a spot light to scene
            LightGeom newLight(SPOTLIGHT);
            glm::vec3 emission = glm::vec3(_light.color[0], _light.color[1], _light.color[2]) * (float)_light.intensity;
            glm::vec3 position = glm::vec3(lightinstance.world[3]);
            glm::vec3 dir = glm::normalize(glm::mat3(lightinstance.world) * glm::vec3(0, 0, -1));
            newLight.emission = emission;
            newLight.position = position;
            newLight.u = dir;
            newLight.innerAngle = _light.spot.innerConeAngle;  // already in radians
            newLight.outerAngle = _light.spot.outerConeAngle;
            if (_light.extras.Has("radius")) {
                newLight.radius = fmax(_light.extras.Get("radius").Get<double>(), 0.01f);
            }
            else {
                newLight.radius = 0.5f;
            }
            newLight.emission /= PI * newLight.radius * newLight.radius;
            scene->lightgeoms.push_back(newLight);
        }
        else if (_light.type == "directional") {
            // add a directional light to scene
            LightGeom newLight(DIRECTIONALLIGHT);
            glm::vec3 emission = glm::vec3(_light.color[0], _light.color[1], _light.color[2]) * (float)_light.intensity;
            /*glm::vec3 position = glm::vec3(lightinstance.world[3]);*/
            glm::vec3 dir = glm::normalize(glm::mat3(lightinstance.world) * glm::vec3(0, 0, -1));
            newLight.emission = emission;
            newLight.position = dir;
            if (_light.extras.Has("alpha")) {
                newLight.radius = _light.extras.Get("alpha").Get<double>() * PI / 180;
            }
            else {
                newLight.radius = 0.265f * PI / 180;
            }
            scene->lightgeoms.push_back(newLight);
        }
    }
}

void Scene::loadFromGLTF(const std::string& fileName, const glm::mat4& inputTransform, bool loadCam, bool loadLgt, int overridematIndex)
{
    std::string ext = fileName.substr(fileName.find_last_of(".") + 1);

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string error, warning;

    loader.SetImageLoader(customLoadImageData, nullptr);

    bool success = false;
    if (ext == "gltf") {
        success = loader.LoadASCIIFromFile(&model, &error, &warning, fileName);
    }
    else {
        success = loader.LoadBinaryFromFile(&model, &error, &warning, fileName);
    }
    if (!success) {
        std::cout << "failed to load model" << error << std::endl;
        exit(-1);
    }

    std::vector<MeshInstance> meshinstances;
    std::vector<CameraInstance> camerainstances;
    std::vector<LightInstance> lightinstances;
    CollectInstancesOneScene(model, inputTransform, meshinstances, camerainstances, lightinstances);
    assert(meshinstances.size() > 0, "should have at least 1 mesh instance in the scene");

    MeshPrims.clear(); // clear meshes, prevent reading previous mesh idx from other gltf files
    loadMeshes(this, model);
    loadMeshInstances(this, meshinstances, overridematIndex);

    std::vector<bool> isNormal;
    isNormal.resize(model.textures.size());
    loadMaterials(this, model, isNormal);
    loadTextures(this, model, isNormal);

    if (loadCam) {
        loadCamera(this, model, camerainstances);
    }
    if (loadLgt) {
        loadLights(this, model, lightinstances);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    size_t slashpos = jsonName.find_last_of('/');
    std::string baseDir;
    if (slashpos != std::string::npos) {
        baseDir = jsonName.substr(0, slashpos + 1);
    }
    else {
        baseDir = "";
    }

    std::ifstream f(jsonName);
    if (!f.is_open()) {
        std::cerr << "File open failed: " << jsonName << std::endl;
        exit(-1);
    }
    json data = json::parse(f);

    // Camera ans State settings
    Camera& camera = state.camera;
    RenderState& state = this->state;
    {
        const auto& cameraData = data["Camera"];
        // state 
        state.iterations = cameraData["ITERATIONS"];
        state.traceDepth = cameraData["DEPTH"];
        state.imageName = cameraData["FILE"];
        // resolution
        camera.resolution.x = cameraData["RES"][0];
        camera.resolution.y = cameraData["RES"][1];
        // maximum resolution: 15360*8640
        if (camera.resolution.x * camera.resolution.y > (1 << 27)) {
            std::cerr << "Maximum Resolution cannot exceed 15360*8640" << '\n';
        }
        // position and rotation
        const auto& pos = cameraData["EYE"];
        const auto& lookat = cameraData["LOOKAT"];
        const auto& up = cameraData["UP"];
        camera.position = glm::vec3(pos[0], pos[1], pos[2]);
        camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
        camera.up = glm::vec3(up[0], up[1], up[2]);
        camera.view = glm::normalize(camera.lookAt - camera.position);
        camera.right = glm::normalize(glm::cross(camera.view, camera.up));
        // calculate fov based on resolution
        float fovy = cameraData["FOVY"];
        float yscaled = tan(fovy * 0.5f * (PI / 180));
        float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
        float fovx = (atan(xscaled) * 180) / PI;
        camera.fov = glm::vec2(fovx, fovy);
        camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
            2 * yscaled / (float)camera.resolution.y);
        // dof related params
        if (cameraData.contains("FOCALDISTANCE")) {
            camera.focalDistance = cameraData["FOCALDISTANCE"];
        }
        else {
            camera.focalDistance = 10.f;
        }
        if (cameraData.contains("LENRADIUS")) {
            camera.lenRadius = cameraData["LENRADIUS"];
        }
        else {
            camera.lenRadius = 0.f;
        }
        camera.autoFocus = true;
    }

    // Matetials
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial;
        // handle materials loading
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = srgbToLinear(glm::vec3(col[0], col[1], col[2]));
            newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = srgbToLinear(glm::vec3(col[0], col[1], col[2]));
            newMaterial.transmission = p.value("TRANSMISSION", 0.f);
            newMaterial.ior = p.value("IOR", 1.5f);
            newMaterial.type = SPECULAR;
        }
        else if (p["TYPE"] == "Disney")
        {
            const auto& col = p["RGB"];
            newMaterial.color = srgbToLinear(glm::vec3(col[0], col[1], col[2]));
            if (p.contains("EMISSION")) {
                const auto& emi = p["EMISSION"];
                newMaterial.emission = glm::vec3(emi[0], emi[1], emi[2]);
            }
            newMaterial.roughness = glm::max(p.value("ROUGHNESS", 0.2f), 0.001f);
            newMaterial.metallic = p.value("METALLIC", 0.f);
            newMaterial.transmission = p.value("TRANSMISSION", 0.f);
            newMaterial.ior = p.value("IOR", 1.5f);
            newMaterial.clearcoat = p.value("CLEARCOAT", 0.f);
            float coatGlossiness = p.value("CLEARCOAT_GLOSS", 1.f);
            newMaterial.coatroughness = glm::mix(0.1f, 0.001f, coatGlossiness);
            newMaterial.subsurface = p.value("SUBSURFACE", 0.f);
            newMaterial.type = DISNEY;
        }

        newMaterial.linecolor = glm::vec3(-2.f);
        if (p.contains("LINECOLOR")) {
            const auto& linecolor = p["LINECOLOR"];
            newMaterial.linecolor = srgbToLinear(glm::vec3(linecolor[0], linecolor[1], linecolor[2]));
        }
        newMaterial.toonshading = p.value("TOON", false);

        MatNameToID[name] = materials.size();
        materials.push_back(newMaterial);
    }

    // Lights
    const auto& lightsData = data["Lights"];
    for (const auto& p : lightsData)
    {
        const auto& type = p["TYPE"];
        if (type == "sphere")
        {
            LightGeom newLight(SPHERELIGHT);
            const auto& emission = p["EMISSION"];
            const auto& position = p["POSITION"];
            const auto& radius = p["RADIUS"];
            newLight.emission = glm::vec3(emission[0], emission[1], emission[2]);
            newLight.position = glm::vec3(position[0], position[1], position[2]);
            newLight.radius = radius;
            lightgeoms.push_back(newLight);
        }
        else if (type == "rect")
        {
            LightGeom newLight(RECTLIGHT);
            const auto& emission = p["EMISSION"];
            const auto& position = p["POSITION"];
            const auto& u = p["EDGE1"];
            const auto& v = p["EDGE2"];
            newLight.emission = glm::vec3(emission[0], emission[1], emission[2]);
            newLight.position = glm::vec3(position[0], position[1], position[2]);
            newLight.u = glm::vec3(u[0], u[1], u[2]);
            newLight.v = glm::vec3(v[0], v[1], v[2]);
            lightgeoms.push_back(newLight);
        }
        else if (type == "directional")
        {
            LightGeom newLight(DIRECTIONALLIGHT);
            const auto& emission = p["EMISSION"];
            const auto& position = p["POSITION"];
            newLight.emission = glm::vec3(emission[0], emission[1], emission[2]);
            newLight.position = glm::normalize(glm::vec3(position[0], position[1], position[2]));
            newLight.radius = p.value("ALPHA", 0.265f) * PI / 180;  // half angle range to create soft shadow for directional light, sun is 0.265 degrees
            lightgeoms.push_back(newLight);
        }
    }

    // Objects
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        if (type == "cube") 
        {
            int vertBase = vertPos.size();

            const auto& center = p["POSITION"];
            const auto& size = p["SIZE"];
            int schannel = p.value("SCHANNEL", 0);
            glm::vec3 centervec(center[0], center[1], center[2]);
            glm::vec3 sizevec(size[0], size[1], size[2]);
            std::vector<glm::vec3> posarray;
            for (int z = -1; z <= 1; z += 2) {
                for (int y = -1; y <= 1; y += 2) {
                    for (int x = -1; x <= 1; x += 2) {
                        glm::vec3 pos = centervec + glm::vec3(x, y, z) * sizevec * 0.5f;
                        posarray.push_back(pos);
                    }
                }
            }

            glm::ivec3 boxTriangles[12] = {
                glm::ivec3(0, 2, 1),
                glm::ivec3(1, 2, 3),
                glm::ivec3(1, 3, 5),
                glm::ivec3(5, 3, 7),
                glm::ivec3(5, 7, 4),
                glm::ivec3(4, 7, 6),
                glm::ivec3(4, 6, 0),
                glm::ivec3(0, 6, 2),
                glm::ivec3(7, 3, 6),
                glm::ivec3(6, 3, 2),
                glm::ivec3(1, 5, 0),
                glm::ivec3(0, 5, 4)
            };
            glm::vec3 boxNormals[12] = {
                glm::vec3(0, 0, -1),
                glm::vec3(0, 0, -1),
                glm::vec3(1, 0, 0),
                glm::vec3(1, 0, 0),
                glm::vec3(0, 0, 1),
                glm::vec3(0, 0, 1),
                glm::vec3(-1, 0, 0),
                glm::vec3(-1, 0, 0),
                glm::vec3(0, 1, 0),
                glm::vec3(0, 1, 0),
                glm::vec3(0, -1, 0),
                glm::vec3(0, -1, 0)
            };
            glm::vec2 faceUVs[6] = {
                glm::vec2(1, 0),
                glm::vec2(1, 1),
                glm::vec2(0, 0),
                glm::vec2(0, 0),
                glm::vec2(1, 1),
                glm::vec2(0, 1)
            };
            for (int i = 0; i < 12; ++i) {
                Geom newGeom(TRIANGLE);

                newGeom.vertIds = glm::ivec3(vertBase + 3 * i) + glm::ivec3(0, 1, 2);
                for (int j = 0; j < 3; ++j) {
                    vertPos.push_back(posarray[boxTriangles[i][j]]);
                    vertNor.push_back(boxNormals[i]);
                    vertUV.push_back(glm::vec4(faceUVs[(i & 1) + j], faceUVs[(i & 1) + j]));
                    vertSchannel.push_back(static_cast<char>(schannel));
                }

                newGeom.materialid = MatNameToID[p["MATERIAL"]];

                geoms.push_back(newGeom);
            }
        }
        else if (type == "sphere")
        {
            Geom newGeom(SPHERE);

            const auto& center = p["POSITION"];
            const auto& radius = p["RADIUS"];

            newGeom.center = glm::vec3(center[0], center[1], center[2]);
            newGeom.radius = radius;

            newGeom.materialid = MatNameToID[p["MATERIAL"]];

            geoms.push_back(newGeom);
        }
        else if (type == "gltf")
        {
            const auto& mesh_path = p["PATH"];
            std::string fullmeshpath = baseDir + mesh_path.get<std::string>();

            glm::vec3 translation(0.f), rotation(0.f), scalevec(1.f);
            if (p.contains("TRANS")) {
                const auto& trans = p["TRANS"];
                translation = glm::vec3(trans[0], trans[1], trans[2]);
            }
            if (p.contains("ROTAT")) {
                const auto& rotat = p["ROTAT"];
                rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            }
            if (p.contains("SCALE")) {
                const auto& scale = p["SCALE"];
                scalevec = glm::vec3(scale[0], scale[1], scale[2]);
            }

            int overridematIndex = -1;
            if (p.contains("OVERRIDE_MATERIAL")) {
                overridematIndex = MatNameToID[p["OVERRIDE_MATERIAL"]];
                assert(overridematIndex >= 0);
            }
            
            glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scalevec);
            bool loadCam = p.value("OVERRIDE_CAMERA", false);
            bool loadLgt = p.value("LOAD_LIGHT", false);
            loadFromGLTF(fullmeshpath, transform, loadCam, loadLgt, overridematIndex);
        }
    }

    // Create BVH
    std::vector<std::shared_ptr<Primitive>> primitives;
    for (int i = 0; i < geoms.size(); ++i) {
        const Geom& geom = geoms[i];
        if (geom.type == SPHERE) {
            primitives.push_back(std::make_shared<Primitive>(i, geom.center, geom.radius));
        }
        else {
            primitives.push_back(std::make_shared<Primitive>(i, vertPos[geom.vertIds[0]], vertPos[geom.vertIds[1]], vertPos[geom.vertIds[2]]));
        }
    }
    bvhAccel = CreateBVHAccelerator(primitives, 1);
#if PT_LINE_RENDER
    float avgLineDistance = 0.f;
    std::vector<std::shared_ptr<Primitive>> extend_primitives;
    for (int i = 0; i < geoms.size(); ++i) {
        const Geom& geom = geoms[i];
        // only put line rendering stuff into extend BVH
        if (geom.materialid >= 0 && materials[geom.materialid].linecolor.x >= 0.f) {
            if (geom.type == SPHERE) {
                extend_primitives.push_back(std::make_shared<Primitive>(i, geom.center, geom.radius + PT_LINE_MAXWIDTH));
                avgLineDistance += glm::length(geom.center - camera.position);
            }
            else {
                Bounds3f oriBound(vertPos[geom.vertIds[0]], vertPos[geom.vertIds[1]]);
                oriBound = Union(oriBound, vertPos[geom.vertIds[2]]);
                Bounds3f extendBound(vertPos[geom.vertIds[0]] + (float)PT_LINE_MAXWIDTH * vertNor[geom.vertIds[0]],
                    vertPos[geom.vertIds[1]] + (float)PT_LINE_MAXWIDTH * vertNor[geom.vertIds[1]]);
                extendBound = Union(extendBound, vertPos[geom.vertIds[2]] + (float)PT_LINE_MAXWIDTH * vertNor[geom.vertIds[2]]);
                extend_primitives.push_back(std::make_shared<Primitive>(i, Union(oriBound, extendBound)));
                avgLineDistance += glm::length(oriBound.Center() - camera.position);
            }
        }
    }
    if (extend_primitives.size() > 0) {
        avgLineDistance /= extend_primitives.size();
    }
    extend_bvhAccel = CreateBVHAccelerator(extend_primitives, 1);
    approxLineDist = avgLineDistance * 1.4f + 0.5f;
#endif // PT_LINE_RENDER
    // Env Map
    if (data.contains("EnvMap")) {
        const auto& EnvMapData = data["EnvMap"];
        const auto& envmap_path = EnvMapData["PATH"];
        std::string fullenvpath = baseDir + envmap_path.get<std::string>();
        envMap.loadToCPU(fullenvpath);
    }

    // Postprocess
    const auto& postprocessData = data["Postprocess"];
    {
        postparams.viewTrans = ColorGradingParams::ViewTransform::NONE;
        if (postprocessData.contains("VIEWTRANSFORM")) {
            if (postprocessData["VIEWTRANSFORM"] == "ACES") {
                postparams.viewTrans = ColorGradingParams::ViewTransform::ACES;
            }
            else if (postprocessData["VIEWTRANSFORM"] == "REINHARD_L") {
                postparams.viewTrans = ColorGradingParams::ViewTransform::REINHARD_L;
            }
        }
        
        postparams.exposureEV = 0.f;
        if (postprocessData.contains("EXPOSURE")) {
            postparams.exposureEV = postprocessData["EXPOSURE"];
        }

        postparams.temperature = 0.f;
        if (postprocessData.contains("TEMPERATURE")) {
            postparams.temperature = glm::clamp((float)(postprocessData["TEMPERATURE"]), -1.f, 1.f);
        }
        
        postparams.tint = 0.f;
        if (postprocessData.contains("TINT")) {
            postparams.tint = glm::clamp((float)(postprocessData["TINT"]), -1.f, 1.f);
        }

        postparams.saturation = 0.f;
        if (postprocessData.contains("SATURATION")) {
            postparams.saturation = glm::clamp((float)(postprocessData["SATURATION"]), -1.f, 1.f);
        }

        postparams.vibrance = 0.f;
        if (postprocessData.contains("VIBRANCE")) {
            postparams.vibrance = glm::clamp((float)(postprocessData["VIBRANCE"]), 0.f, 1.f);
        }

        postparams.contrast = 0.f;
        if (postprocessData.contains("CONTRAST")) {
            postparams.contrast = glm::clamp((float)(postprocessData["CONTRAST"]), -1.f, 1.f);
        }

        postparams.contrastPivot = 0.18f;
        if (postprocessData.contains("CONTRAST_PIVOT")) {
            postparams.contrastPivot = glm::clamp((float)(postprocessData["CONTRAST_PIVOT"]), 0.f, 1.f);
        }

        if (postprocessData.contains("CDLSLOPE")) {
            const auto& cdlSlope = postprocessData["CDLSLOPE"];
            postparams.cdlSlope = glm::vec3(cdlSlope[0], cdlSlope[1], cdlSlope[2]);
        }
        if (postprocessData.contains("CDLOFFSET")) {
            const auto& cdlOffset = postprocessData["CDLOFFSET"];
            postparams.cdlOffset = glm::vec3(cdlOffset[0], cdlOffset[1], cdlOffset[2]);
        }
        if (postprocessData.contains("CDLPOWER")) {
            const auto& cdlPower = postprocessData["CDLPOWER"];
            postparams.cdlPower = glm::vec3(cdlPower[0], cdlPower[1], cdlPower[2]);
        }
    }

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
