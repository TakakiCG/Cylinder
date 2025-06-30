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

    /// newton法で収束しなった回数カウント用
    mutable int exceptionCount; // mutableにしてconst関数内でも変更可能にする
    void printExceptionCount() const; // 新しい関数

    Renderer(const std::vector<Body> &bodies, Camera camera, Color bgColor=Color::Zero());

    double rand() const;

    bool hitScene(const Ray &ray, RayHit &hit) const;

    Image render() const;

//    Image directIlluminationRender(const unsigned int &samples) const;
//    Image _directIlluminationRender_anti_areas(const unsigned int &_samples, const unsigned int &areas_n_samples) const;

/// Kajiya Kay
    Image _directIlluminationRender(const unsigned int &samples) const;

    Image passTracingRender(const unsigned int &samples) const;

    Image passTracingRender_antiAreas(const unsigned int &samples) const;

    Color trace(const Ray &ray, const RayHit &hit) const;

    Color traceMarschner(const Ray &ray, const RayHit &hit) const;

    void diffuseSample(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray) const;

    void diffuseSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray) const;

    double newton_method() const;

    void specularSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, const double & n) const;

    static void marschnerSampleHair(const Ray &in_Ray, const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &axis,
                             Ray &out_ray) ;

    double FresnelFunction(double cos_gammaI, double eta) const;

    static void computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v);

    /// Visualize theta and phi
    struct RaySampleData {
        int    sampleIdx     = 0;    // 何番目のサンプルか
        double incidentTheta = 0.0;  // θ_in  [rad]
        double incidentPhi   = 0.0;
        double outgoingTheta = 0.0;  // θ_out [rad]
        double outgoingPhi   = 0.0;  // φ_out [rad]

        std::string hitType      = "Unknown";   // "Sphere" or "Cylinder"
        std::string scatterType  = "None";      // "Diffuse" or "Specular" (or "None")
        std::string reason      = "N/A";       // why scatterType == None
    };

    mutable std::vector<RaySampleData> g_samples;
    mutable std::mutex                 g_mutex;

    Color trace(const Ray &ray, const RayHit &hit, bool recordAngles, int sampleIdx) const;

    void diffuseSample(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, bool recordAngles, RaySampleData* rec) const;
    void diffuseSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, bool recordAngles, RaySampleData* rec) const;
    void specularSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, const double & n, bool recordAngles, RaySampleData* rec) const;
    void worldToLocalAngles(const Eigen::Vector3d& n, const Eigen::Vector3d& dir, double& theta, double& phi) const;
    static void marschnerSampleHair(const Ray& in_Ray,
                                    const Eigen::Vector3d& incidentPoint,
                                    const Eigen::Vector3d& axis,
                                    Ray& out_Ray,
                                    bool recordAngles = false,
                                    RaySampleData* rec = nullptr);


/// Marschner
//    struct BSDFParams{
//        const double eta = 1.55;
//        const double alpha_R = 0;
//        const double beta_R = 0;
//    };

//    Color fur_bsdf(const Eigen::Vector3d &wi, const Eigen::Vector3d &wo, const BSDFParams &params) const;

//    void marschnerSample(const Ray &in_ray, const Eigen::Vector3d &incidentPoint,
//                         const Eigen::Vector3d &hairDir, const Material &material,
//                         Ray &out_Ray) const;
//
//    Color marschnerShading(const Eigen::Vector3d &V, const Eigen::Vector3d &L,
//                           const Eigen::Vector3d &H, const Material &material) const;


private:
    // Marschnerモデルのヘルパー関数
    double computeLongitudinalScattering(double theta_i, double theta_r, double beta_m) const;
    double computeAzimuthalScattering(double phi, double beta_n) const;
};


#endif //DAY_3_RENDERER_H
