#pragma once

#include "scene.h"
#include "utilities.h"

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtraceClear();
bool pathtraceFocusOnPixel(int px, int py);
void pathtraceGetGBuffer();
void pathtrace(uchar4 *pbo, int maxiter, int iteration);
