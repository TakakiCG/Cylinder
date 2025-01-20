//
// Created by Takaki on 2024/10/28.
//

#include "Cylinder.h"

bool Cylinder::hit(const Ray &ray, RayHit &hit) const {
    /// ２次方程式を解く
    //Eigen::Vector3d oc = ray.org - baseCenter;
    Eigen::Vector3d d = ray.dir - (ray.dir.dot(axis)) * axis;   /// ray.dirからaxis方向成分を引いたもの(axisに垂直)
    Eigen::Vector3d o = (ray.org - baseCenter) - ( (ray.org - baseCenter).dot(axis) ) * axis;

    double a = d.dot(d);
    double b = 2.0 * o.dot(d);
    double c = o.dot(o) - radius * radius;
    double discriminant = b * b - 4 * a * c;

    if (discriminant < 0) return false;

    double t0 = (-b - sqrt(discriminant)) / (2.0 * a);
    double t1 = (-b + sqrt(discriminant)) / (2.0 * a);

    if (t0 > t1) std::swap(t0, t1);

    double y0 = (ray.org - baseCenter).dot(axis) + t0 * ray.dir.dot(axis);
    double y1 = (ray.org - baseCenter).dot(axis) + t1 * ray.dir.dot(axis);

    if (y0 < 0 || y0 > height) {
        if (y1 < 0 || y1 > height) return false;
        t0 = t1;
    }

    if (t0 < 1e-6) return false;

    hit.t = t0;
    hit.point = ray.at(hit.t);
    hit.normal = ((hit.point - baseCenter) - (hit.point - baseCenter).dot(axis) * axis).normalized();

    /// added to get h
//    // ヒット情報を詰める
//    hit.idx = 0; // とりあえず 0 番目のオブジェクトという意味で
//    hit.t = t0;
//    hit.point = ray.at(t0);
//
//    // 円柱の法線ベクトル
//    // (衝突点 - baseCenter) から軸方向成分を抜き、正規化
//    Eigen::Vector3d v = (hit.point - baseCenter);
//    hit.normal = (v - (v.dot(axis)) * axis).normalized();
//
//    // 4) 衝突点の断面中心 C
//    double y_hit = (ray.org - baseCenter).dot(axis) + t0 * (ray.dir.dot(axis));
//    Eigen::Vector3d C = baseCenter + y_hit * axis;
//
//    // 5) レイ (org + t*dir) と、C を通り dir と平行な直線(ℓ(u)=C+u*dir)
//    //    との最短距離を計算:  distance = |(C - org) × dir| / |dir|
//    //    (dir は単位ベクトルなので分母は 1)
//    Eigen::Vector3d OC = C - ray.org;
//    h = (OC.cross(ray.dir)).norm();

    return true;
}