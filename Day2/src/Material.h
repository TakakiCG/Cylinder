//
// Created by kango on 2023/04/14.
//

#ifndef DAY_3_MATERIAL_H
#define DAY_3_MATERIAL_H

#include <utility>

#include "Image.h"

struct Material {
    Color color;
    double kd;  // 拡散反射係数
    double emission;
    /// add
    double ks;  // 鏡面反射係数
    double n;   // 輝き係数

    /// Marschnerモデルのパラメータ
    double eta;      // 屈折率
    Color sigma_a;   // 吸収係数（波長ごとに設定可能なようにColor型）
    double beta_m;   // 縦方向の粗さ
    double beta_n;   // 横方向の粗さ
    double alpha;    // 主なスケール角度

    // シェーディングモデルの選択
    enum ShadingModel {
        KAJIYA_KAY,
        MARSCHNER
    } shadingModel;

//public:
//    Material(Color color, const double &kd, const double &emission=0.0) : color(std::move(color)), kd(kd), emission(emission) {}
//};
//public:
//    Material(Color color, const double &kd, const double &emission=0.0, const double &ks=0.0, const double &n=50.0)
//            : color(std::move(color)), kd(kd), emission(emission), ks(ks), n(n) {}

    Material(Color color, double kd, double emission = 0.0, double ks = 0.0, double n =0.0,
             double eta = 0.0, Color sigma_a = Color(0.2, 0.2, 0.2), double beta_m = 0.0, double beta_n = 0.0,
             ShadingModel shadingModel = KAJIYA_KAY)
            : color(std::move(color)), kd(kd), emission(emission), ks(ks), n(n),
              eta(eta), sigma_a(std::move(sigma_a)), beta_m(beta_m), beta_n(beta_n),
              shadingModel(shadingModel) {}
};

#endif //DAY_3_MATERIAL_H
