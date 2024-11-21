//
// Created by Takaki on 2024/11/19.
//

#ifndef RENDERINGWORKSHOP_BEZIERCURVE_H
#define RENDERINGWORKSHOP_BEZIERCURVE_H


#include "Eigen/Core"

class BezierCurve {
public:
    BezierCurve(const std::vector<Eigen::Vector3d>& controlPoints);
    Eigen::Vector3d evaluate(double t) const;
    Eigen::Vector3d derivative(double t) const;

private:
    std::vector<Eigen::Vector3d> controlPoints;
};


#endif //RENDERINGWORKSHOP_BEZIERCURVE_H
