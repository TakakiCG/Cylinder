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

    return true;
}