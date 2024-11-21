//
// Created by kango on 2023/04/03.
//

#ifndef DAY_3_RENDERER_H
#define DAY_3_RENDERER_H


#include <vector>
#include "Body.h"
#include "Camera.h"
#include <random>
#include "Eigen/Dense"

class Renderer {
public:
    std::vector<Body> bodies;

    Camera camera;
    Color bgColor;

    /// 乱数生成器
    mutable std::mt19937_64 engine;
    mutable std::uniform_real_distribution<> dist;

    Renderer(const std::vector<Body> &bodies, Camera camera, Color bgColor=Color::Zero());

    double rand() const;

    bool hitScene(const Ray &ray, RayHit &hit) const;

    Image render() const;

    Image directIlluminationRender(const unsigned int &samples) const;

    Image _directIlluminationRender(const unsigned int &samples) const;

    void diffuseSample(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray) const;

    void diffuseSampleHair(Ray &in_ray, const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray) const;

    void marschnerSample(const Ray &in_ray, const Eigen::Vector3d &incidentPoint,
                         const Eigen::Vector3d &hairDir, const Material &material,
                         Ray &out_Ray) const;

    static void computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v);

    static void computeLocalFrameHair(const Eigen::Vector3d &u, Eigen::Vector3d &w, Eigen::Vector3d &v);

    Color kajiyaKayShading(const Eigen::Vector3d &V, const Eigen::Vector3d &L,
                           const Eigen::Vector3d &H, const Material &material) const;

    Color marschnerShading(const Eigen::Vector3d &V, const Eigen::Vector3d &L,
                           const Eigen::Vector3d &H, const Material &material) const;

private:
    // Marschnerモデルのヘルパー関数
    double computeLongitudinalScattering(double theta_i, double theta_r, double beta_m) const;
    double computeAzimuthalScattering(double phi, double beta_n) const;

};


#endif //DAY_3_RENDERER_H
