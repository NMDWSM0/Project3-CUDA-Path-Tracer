#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

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

void Scene::loadFromJSON(const std::string& jsonName)
{
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
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.type = DIFFUSE;
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.transmission = p.value("TRANSMISSION", 0.f);
            newMaterial.ior = p.value("IOR", 1.5f);
            newMaterial.type = SPECULAR;
        }
        else if (p["TYPE"] == "Disney")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
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
