#pragma once

#include "sceneStructs.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<LightGeom> lightgeoms;
    std::vector<glm::vec3> vertPos;
    std::vector<glm::vec3> vertNor;
    std::vector<glm::vec2> vertUV;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    Texture envMap;
    RenderState state;
};
