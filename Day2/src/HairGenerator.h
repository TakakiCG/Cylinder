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
};



#endif //RENDERINGWORKSHOP_HAIRGENERATOR_H
