#pragma once

#include "postprocess.h"
#include "sceneStructs.h"
#include "bvh.h"
#include <vector>

class Scene
{
private:
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& fileName, const glm::mat4& inputTransform, bool loadCam, bool loadLgt);
public:
    Scene(std::string filename);

    std::vector<Geom> geoms;
    std::vector<LightGeom> lightgeoms;
    std::vector<glm::vec3> vertPos;
    std::vector<glm::vec3> vertNor;
    std::vector<glm::vec4> vertUV;
    std::vector<char> vertSchannel;
    std::vector<Material> materials;
    std::vector<Texture> textures;
    Texture envMap;
    RenderState state;
    std::shared_ptr<BVHAccel> bvhAccel;
    std::shared_ptr<BVHAccel> extend_bvhAccel;
    float approxLineDist;
    ColorGradingParams postparams;
};
