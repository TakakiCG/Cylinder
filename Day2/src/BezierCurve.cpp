//
// Created by Takaki on 2024/11/19.
//

#include "BezierCurve.h"

BezierCurve::BezierCurve(const std::vector<Eigen::Vector3d>& controlPoints)
        : controlPoints(controlPoints) {}

Eigen::Vector3d BezierCurve::evaluate(double t) const {
    // de Casteljau's algorithm for cubic Bezier curves
    Eigen::Vector3d P0 = controlPoints[0];
    Eigen::Vector3d P1 = controlPoints[1];
    Eigen::Vector3d P2 = controlPoints[2];
    Eigen::Vector3d P3 = controlPoints[3];

    double u = 1 - t;

    return u*u*u*P0 + 3*u*u*t*P1 + 3*u*t*t*P2 + t*t*t*P3;
}

Eigen::Vector3d BezierCurve::derivative(double t) const {
    // Derivative of cubic Bezier curve
    Eigen::Vector3d P0 = controlPoints[0];
    Eigen::Vector3d P1 = controlPoints[1];
    Eigen::Vector3d P2 = controlPoints[2];
    Eigen::Vector3d P3 = controlPoints[3];

    double u = 1 - t;

    return 3*u*u*(P1 - P0) + 6*u*t*(P2 - P1) + 3*t*t*(P3 - P2);
}