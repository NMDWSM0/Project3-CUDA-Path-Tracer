#include "scene.h"
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

static struct MeshInstance {
    int node = -1;
    int mesh = -1;
    glm::mat4 world{ 1.0f };
};

static void DFS(const tinygltf::Model& m, int ni, const glm::mat4& parent, std::vector<MeshInstance>& out) {
    const auto& n = m.nodes[ni];
    glm::mat4 world = parent * LocalOf(n);
    if (n.mesh >= 0) 
        out.push_back({ ni, n.mesh, world });
    for (int c : n.children) 
        DFS(m, c, world, out);
}

static std::vector<MeshInstance> CollectInstancesOneScene(const tinygltf::Model& model) {
    std::vector<MeshInstance> inst;
    const int s = (model.defaultScene >= 0) ? model.defaultScene : 0;
    for (int root : model.scenes[s].nodes) 
        DFS(model, root, glm::mat4(1.0f), inst);
    return inst;
}

static void loadMeshes(Scene* scene, tinygltf::Model& gltfModel, const glm::mat4& inputTransform)
{
    glm::mat3 normalTransform = glm::inverseTranspose(glm::mat3(inputTransform));
    for (int gltfMeshIdx = 0; gltfMeshIdx < gltfModel.meshes.size(); gltfMeshIdx++)
    {
        tinygltf::Mesh gltfMesh = gltfModel.meshes[gltfMeshIdx];

        for (int gltfPrimIdx = 0; gltfPrimIdx < gltfMesh.primitives.size(); gltfPrimIdx++)
        {
            tinygltf::Primitive prim = gltfMesh.primitives[gltfPrimIdx];

            // Skip points and lines
            if (prim.mode != TINYGLTF_MODE_TRIANGLES)
                continue;

            int indicesIndex = prim.indices;
            int positionIndex = -1;
            int normalIndex = -1;
            int uv0Index = -1;

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

            std::vector<glm::vec3> vertices;
            std::vector<glm::vec3> normals;
            std::vector<glm::vec2> uvs;

            // Get vertex data
            for (size_t vertexIndex = 0; vertexIndex < positionAccessor.count; vertexIndex++)
            {
                glm::vec3 vertex, normal;
                glm::vec2 uv;

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
                    memcpy(&uv, address, sizeof(glm::vec2));
                }

                vertices.push_back(glm::vec3(inputTransform * glm::vec4(vertex, 1.0f)));
                normals.push_back(glm::normalize(normalTransform * normal));
                uvs.push_back(uv);
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

            // push geometrys, be careful at mat/vert index offset
            int vertBase = scene->vertPos.size();
            int sceneMatIdx = prim.material + scene->materials.size();
            for (int tid = 0; tid * 3 < indices.size(); ++tid) {
                Geom newGeom(TRIANGLE);
                newGeom.vertIds = glm::ivec3(indices[3 * tid], indices[3 * tid + 1], indices[3 * tid + 2]) + glm::ivec3(vertBase);
                newGeom.materialid = sceneMatIdx;
                scene->geoms.push_back(newGeom);
            }

            // push pos, nor and uv
            scene->vertPos.insert(scene->vertPos.end(), vertices.begin(), vertices.end());
            scene->vertNor.insert(scene->vertNor.end(), normals.begin(), normals.end());
            scene->vertUV.insert(scene->vertUV.end(), uvs.begin(), uvs.end());
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

void loadTextures(Scene* scene, tinygltf::Model& gltfModel)
{
    for (size_t i = 0; i < gltfModel.textures.size(); ++i)
    {
        tinygltf::Texture& gltfTex = gltfModel.textures[i];
        tinygltf::Image& image = gltfModel.images[gltfTex.source];
        Texture texture;
        texture.loadToCPU(image.image.data(), image.width, image.height, image.component);
        scene->textures.push_back(texture);
    }
}

void loadMaterials(Scene* scene, tinygltf::Model& gltfModel)
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
            material.baseColorTexId = pbr.baseColorTexture.index + sceneTexIdx;

        // Emission
        material.emission = glm::vec3((float)gltfMaterial.emissiveFactor[0], (float)gltfMaterial.emissiveFactor[1], (float)gltfMaterial.emissiveFactor[2]);
        if (gltfMaterial.emissiveTexture.index > -1)
            material.emissionmapTexId = gltfMaterial.emissiveTexture.index + sceneTexIdx;

        // Roughness and Metallic
        material.roughness = (float)pbr.roughnessFactor;
        material.metallic = (float)pbr.metallicFactor;
        if (pbr.metallicRoughnessTexture.index > -1)
            material.metallicRoughnessTexId = pbr.metallicRoughnessTexture.index + sceneTexIdx;

        // Normal Map
        material.normalmapTexId = gltfMaterial.normalTexture.index + sceneTexIdx;

        // KHR_materials_transmission
        material.transmission = 0.f;
        if (gltfMaterial.extensions.find("KHR_materials_transmission") != gltfMaterial.extensions.end())
        {
            const auto& ext = gltfMaterial.extensions.at("KHR_materials_transmission");
            if (ext.Has("transmissionFactor"))
                material.transmission = (float)(ext.Get("transmissionFactor").Get<double>());
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
        if (gltfMaterial.extensions.find("KHR_materials_emissive_strength") != gltfMaterial.extensions.end())
        {
            const auto& ext = gltfMaterial.extensions.at("KHR_materials_emissive_strength");
            if (ext.Has("emissiveStrength"))
                material.emission *= (float)(ext.Get("emissiveStrength").Get<double>());
        }

        // my_subsurface
        material.subsurface = 0.f;
        if (gltfMaterial.extras.Has("my_subsurface")) {
            material.subsurface = (float)(gltfMaterial.extras.Get("my_subsurface").Get<double>());
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

void Scene::loadFromGLTF(const std::string& fileName, const glm::mat4& inputTransform)
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

    auto instance = CollectInstancesOneScene(model);
    assert(instance.size() > 0, "should have at least 1 instance in the scene");
    glm::mat4 instTransform = instance[0].world;

    loadMeshes(this, model, inputTransform * instTransform);
    loadMaterials(this, model);
    loadTextures(this, model);
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
    json data = json::parse(f);

    // Matetials
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
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
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
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
            newLight.position = glm::vec3(position[0], position[1], position[2]);
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
                    vertUV.push_back(faceUVs[(i & 1) + j]);
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
        else if (type == "mesh")
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
            
            glm::mat4 transform = utilityCore::buildTransformationMatrix(translation, rotation, scalevec);

            loadFromGLTF(fullmeshpath, transform);
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

    // Env Map
    if (data.contains("EnvMap")) {
        const auto& EnvMapData = data["EnvMap"];
        const auto& envmap_path = EnvMapData["PATH"];
        std::string fullenvpath = baseDir + envmap_path.get<std::string>();
        envMap.loadToCPU(fullenvpath);
    }

    // Camera ans State settings
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    // maximum resolution: 15360*8640
    if (camera.resolution.x * camera.resolution.y > (1 << 27)) {
        std::cerr << "Maximum Resolution cannot exceed 15360*8640" << '\n';
    }
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * 0.5f * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
