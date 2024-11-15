//
// Created by kango on 2023/04/14.
//

#ifndef DAY_3_MATERIAL_H
#define DAY_3_MATERIAL_H

#include <utility>

#include "Image.h"

struct Material {
    Color color;
    double kd;
    double emission;
    double refraction;
    double absorption;


public:
    Material(Color color, const double &kd, const double &emission=0.0) : color(std::move(color)), kd(kd), emission(emission), refraction(refraction), absorption(
            absorption) {}
};

#endif //DAY_3_MATERIAL_H
