//
// Created by kango on 2023/04/03.
//

#ifndef DAY_3_BODY_H
#define DAY_3_BODY_H


#include "Sphere.h"
#include "Cylinder.h"
#include "Material.h"

/// シーン内の物体を表す構造体
struct Body {
    Sphere sphere;
    Cylinder cylinder;
    Material material;

    Body(Sphere sphere, Material material) : sphere(std::move(sphere)), material(std::move(material)), type(Type::Sphere) {}
    Body(Cylinder cylinder, Material material) : cylinder(std::move(cylinder)), material((std::move(material))), type(Type::Cylinder) {}

    bool hit(const Ray &ray, RayHit &hit) const {
        if(type == Type::Sphere) {
            return sphere.hit(ray, hit);
        } else if (type == Type::Cylinder) {
            return cylinder.hit(ray, hit);
        } else {
            return false;
        }
    }

//    /// 球との衝突判定を行う関数
//    bool hitSphere(const Ray &ray, RayHit &hit) const {
//        return sphere.hit(ray, hit);
//    }
//
//    /// add
//    /// 円柱との衝突判定を行う関数
//    bool hitCylinder(const Ray &ray, RayHit &hit) const {
//        return cylinder.hit(ray, hit);
//    }

    /// 自己発光成分を取得する関数
    Eigen::Vector3d getEmission() const {
        return material.emission * material.color;
    }

    /// 拡散反射率成分を取得する関数
    Eigen::Vector3d getKd() const {
        return material.kd * material.color;
    }

    /// 球の法線ベクトルを取得する関数
    Eigen::Vector3d getNormalSphere(const Eigen::Vector3d &p) const {
        return  (p - sphere.center).normalized();
    }

    /// add
    /// 円柱の法線ベクトルを取得する関数
//    Eigen::Vector3d getNormalCylinder(const Eigen::Vector3d &p) const {
//        Eigen::Vector3d toPoint = p - cylinder.baseCenter;
//        Eigen::Vector3d projection = toPoint.dot(cylinder.axis) * cylinder.axis;
//        return (toPoint - projection).normalized();
//    }

    /// 光源であるかどうかを判定する関数
    bool isLight() const {
        return material.emission > 0.0;
    }

public:
    enum Type {
        Sphere,
        Cylinder
    };

    Type type;  // オブジェクトの種類を示すフィールド

};

#endif //DAY_3_BODY_H
