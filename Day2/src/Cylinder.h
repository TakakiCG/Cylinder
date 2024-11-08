//
// Created by Takaki on 2024/10/28.
//

#ifndef RENDERINGWORKSHOP_CYLINDER_H
#define RENDERINGWORKSHOP_CYLINDER_H


#include "Eigen/Dense"
#include "Ray.h"

class Cylinder {
public:
    double radius;
    double height;
    Eigen::Vector3d baseCenter;
    Eigen::Vector3d axis;   /// 軸

    Cylinder() = default;
    Cylinder(double radius, double height, Eigen::Vector3d baseCenter, Eigen::Vector3d axis)
            : radius(radius), height(height), baseCenter(baseCenter), axis(axis.normalized()) {}

    bool hit(const Ray &ray, RayHit &hit) const;
};


#endif //RENDERINGWORKSHOP_CYLINDER_H
