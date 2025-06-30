//
// Created by Takaki on 2024/11/19.
//

#include "HairGenerator.h"
#include <random>

std::vector<Body> HairGenerator::generateHairs(int numHairs, int numSegments, double hairRadius, const Eigen::Vector3d& headCenter, double headRadius) {
    std::vector<Body> hairs;
    std::mt19937 rng(0); // 乱数生成器
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::uniform_real_distribution<double> angleDist(0, 2 * EIGEN_PI);
    std::uniform_real_distribution<double> lengthDist(7.0, 10.0); // 髪の毛の長さをランダム化

    for (int i = 0; i < numHairs; ++i) {
        // 頭の表面上のランダムな位置を計算
        double theta = acos(dist(rng)); // 0 ~ π
        double phi = angleDist(rng);    // 0 ~ 2π

        // 頭の後ろ側のみ
        if (theta > EIGEN_PI / 2) {
            --i; // 条件を満たさないので再試行
            continue;
        }

        // 頭の表面上のポイント（髪の毛の根元）
        Eigen::Vector3d P0 = headCenter + headRadius * Eigen::Vector3d(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
        );

        // 髪の毛の方向（頭の法線方向）
        Eigen::Vector3d normal = (P0 - headCenter).normalized();

        // 髪の毛の長さ
        double hairLength = lengthDist(rng);

        // 制御点の設定
        std::vector<Eigen::Vector3d> controlPoints(4);
        // 根元
        controlPoints[0] = P0;

        // P1: 頭のすぐ後ろ
        controlPoints[1] = P0 + normal * 2.0 + Eigen::Vector3d(0, -1, 0) * 1.0;

        // P2: 背中に沿って垂れ下がる中間ポイント
        controlPoints[2] = P0 + Eigen::Vector3d(0, -1, 0) * (hairLength * 0.5) + Eigen::Vector3d(dist(rng), 0, dist(rng)) * 2.0;

        // P3: 髪の毛の先端
        controlPoints[3] = P0 + Eigen::Vector3d(0, -1, 0) * hairLength + Eigen::Vector3d(dist(rng), 0, dist(rng)) * 2.0;

        // ベジェ曲線の作成
        BezierCurve curve(controlPoints);

        // ベジェ曲線をセグメントに分割して円柱として追加
        double step = 1.0 / numSegments;
        Eigen::Vector3d prevPoint = curve.evaluate(0.0);
        for (int s = 1; s <= numSegments; ++s) {
            double t = s * step;
            Eigen::Vector3d currentPoint = curve.evaluate(t);
            Eigen::Vector3d segmentDir = currentPoint - prevPoint;
            double segmentLength = segmentDir.norm();
            if (segmentLength > 1e-6) {
                segmentDir.normalize();
                Cylinder cylinder(hairRadius, segmentLength, prevPoint, segmentDir);    // 黒茶#25160E
                Material hairMaterial(Color(0.2, 0.1, 0.05), 0.2, 0.0, 0.7, 150.0,
                                      1.55, Color(0.2, 0.1, 0.05), 0.3, 0.3, Material::KAJIYA_KAY);
//                Cylinder cylinder(hairRadius, segmentLength, prevPoint, segmentDir);
//                Material hairMaterial(Color(0.005, 0.005, 0.005), 0.2, 0.0, 0.7, 300.0,
//                                      1.55, Color(0.2, 0.1, 0.05), 0.3, 0.3, Material::KAJIYA_KAY); // black
                hairs.emplace_back(cylinder, hairMaterial);
            }
            prevPoint = currentPoint;
        }
    }

    return hairs;
}

std::vector<Body> HairGenerator::generateStraightHairsInLine(int numHairs, double hairRadius, const Eigen::Vector3d& startPoint, double hairLength) {
    std::vector<Body> hairs;

    // 髪の毛は全て下向き
    Eigen::Vector3d direction(0, -1, 0);
    direction.normalize(); // 念のため正規化

    // マテリアル
    Material hairMaterial(
            //Color(1.0, 1.0, 1.0),
            //codeToColor("#f9c89b"),
            Color(0.2, 0.1, 0.05), // base color
            0.4,   // ambient
            0.0,   // diffuse
            0.5,   // specular
            150.0, // shininess
            1.55,  // index of refraction
            Color(0.2, 0.1, 0.05), // secondary color
            0.3,   // Kajiya-Kay parameters
            0.3,
            Material::KAJIYA_KAY
    );

    for (int i = 0; i < numHairs; ++i) {
        // x方向に髪の半径2倍ずつオフセットして配置
        Eigen::Vector3d topPosition = startPoint + Eigen::Vector3d(i * (2.0 * hairRadius), 0.0, 0.0);

        // Cylinderは、中心点と方向ベクトル、および半径・長さを用いて定義
        // ここでは topPosition を円柱の上端とし、directionにhairLengthだけ伸ばした円柱を作る
        // Cylinderコンストラクタは (半径, 長さ, 起点, 方向) と仮定する。
        Cylinder cylinder(hairRadius, hairLength, topPosition, direction);

        hairs.emplace_back(cylinder, hairMaterial);
    }

    return hairs;
}