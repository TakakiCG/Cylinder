//
// Created by Takaki on 2024/11/19.
//

#ifndef RENDERINGWORKSHOP_HAIRGENERATOR_H
#define RENDERINGWORKSHOP_HAIRGENERATOR_H


#include "Body.h"
#include "BezierCurve.h"
#include <vector>

class HairGenerator {
public:
    static std::vector<Body> generateHairs(int numHairs, int numSegments, double hairRadius, const Eigen::Vector3d& headCenter, double headRadius);
    static std::vector<Body> generateStraightHairsInLine(int numHairs, double hairRadius, const Eigen::Vector3d& startPoint, double hairLength);
};



#endif //RENDERINGWORKSHOP_HAIRGENERATOR_H
