//
// Created by kango on 2023/04/03.
//

#ifndef DAY_3_BODY_H
#define DAY_3_BODY_H


#include "Sphere.h"
#include "Cylinder.h"
#include "Material.h"

struct Body {
    Sphere sphere;
    Cylinder cylinder;
    Material material;

    Body(Sphere sphere, Material material) : sphere(std::move(sphere)), material(std::move(material)) {}
    Body(Cylinder cylinder, Material material) : cylinder(std::move(cylinder)), material((std::move(material))) {}

    bool hitSphere(const Ray &ray, RayHit &hit) const {
        return sphere.hit(ray, hit);
    }

    /// add
    bool hitCylinder(const Ray &ray, RayHit &hit) const {
        return cylinder.hit(ray, hit);
    }

    Eigen::Vector3d getEmission() const {
        return material.emission * material.color;
    }

    Eigen::Vector3d getKd() const {
        return material.kd * material.color;
    }

    Eigen::Vector3d getNormalSphere(const Eigen::Vector3d &p) const {
        return  (p - sphere.center).normalized();
    }

    /// add
    Eigen::Vector3d getNormalCylinder(const Eigen::Vector3d &p) const {
        Eigen::Vector3d toPoint = p - cylinder.baseCenter;
        Eigen::Vector3d projection = toPoint.dot(cylinder.axis) * cylinder.axis;
        return (toPoint - projection).normalized();
    }

    bool isLight() const {
        return material.emission > 0.0;
    }
};

#endif //DAY_3_BODY_H
