//
// Created by kango on 2023/03/22.
//

#ifndef DAY_2_RAY_H
#define DAY_2_RAY_H

#include <utility>
#include <iostream>

#include "Eigen/Dense"

/// レイを表現する構造体
struct Ray {
    Eigen::Vector3d org;    // レイの始点（原点）
    Eigen::Vector3d dir;    // レイの方向ベクトル

    Ray() = default;

    // 始点と方向を指定するコンストラクタ
    Ray(const Eigen::Vector3d &org, const Eigen::Vector3d &dir) : org(org), dir(dir.normalized()) {}

    // パラメータtにおける位置を計算
    Eigen::Vector3d at(const double &t) const {
        return org + t * dir;
    }
};

/// レイが物体にヒットした際の情報を格納する構造体
struct RayHit {
    int idx = -1;   // ヒットした物体のインデックス（-1はヒットなしを表す）
    double t = 0.0; // レイがヒットした距離
    Eigen::Vector3d point = Eigen::Vector3d::Zero();    // ヒットした点の座標
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();   // ヒットした点での法線ベクトル

    RayHit() = default;

    // レイが物体にヒットしているかを判定
    bool isHit() const {
        return idx != -1;   // インデックスが-1でなければヒット
    }

    // ヒット情報を出力する関数
    void show() const {
        std::cout << "idx:\t" << idx << std::endl;
        std::cout << "t:\t" << t << std::endl;
        std::cout << "point:\t" << point.transpose() << std::endl;
        std::cout << "normal:\t" << normal.transpose() << std::endl;
    }
};

#endif //DAY_2_RAY_H
