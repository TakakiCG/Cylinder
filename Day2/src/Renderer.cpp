//
// Created by kango on 2023/04/03.
//

#include "Renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

/// シーン内の物体、カメラ、背景色を初期化する
Renderer::Renderer(const std::vector<Body> &bodies, Camera camera, Color bgColor)
        : bodies(bodies), camera(std::move(camera)), bgColor(std::move(bgColor)), engine(0), dist(0, 1) {
}

/// 乱数生成：0から1の範囲で乱数を返す
double Renderer::rand() const {
    return dist(engine);
}

/**
 * \b シーン内に存在するBodyのうちレイにhitするものを探す
 * @param ray レイ
 * @param hit hitした物体の情報を格納するRayHit構造体
 * @return 何かしらのBodyにhitしたかどうかの真偽値
 */
bool Renderer::hitScene(const Ray &ray, RayHit &hit) const {
    /// hitするBodyのうち最小距離のものを探す
    hit.t = DBL_MAX;
    hit.idx = -1;
    for (int i = 0; i < bodies.size(); ++i) {
        RayHit _hit;
        if (bodies[i].hit(ray, _hit) && _hit.t < hit.t) {
            hit.t = _hit.t;
            hit.idx = i;
            hit.point = _hit.point;
            hit.normal = _hit.normal;
        }
    }

    return hit.idx != -1;
}

Image Renderer::render() const {
    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());
    /// フィルム上のピクセル全てに向けてレイを飛ばす
    for (int p_y = 0; p_y < image.height; p_y++) {
        for (int p_x = 0; p_x < image.width; p_x++) {
            const int p_idx = p_y * image.width + p_x;
            Color color;
            Ray ray;
            RayHit hit;
            camera.filmView(p_x, p_y, ray);

            /// レイを飛ばし、Bodyに当たったらその色を格納する\n
            /// 当たらなければ、背景色を返す
            color = hitScene(ray, hit) ? bodies[hit.idx].material.color : bgColor;
            image.pixels[p_idx] = color;
        }
    }

    return image;
}

Image Renderer::directIlluminationRender(const unsigned int &samples) const {
    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());
    /// フィルム上のピクセル全てに向けてレイを飛ばす
#pragma omp parallel for
    for (int p_y = 0; p_y < image.height; p_y++) {
        for (int p_x = 0; p_x < image.width; p_x++) {
            const int p_idx = p_y * image.width + p_x;
            Ray ray;
            RayHit hit;
            camera.filmView(p_x, p_y, ray);

            if (hitScene(ray, hit)) {
                Color reflectRadiance = Color::Zero();
                for (int i = 0; i < samples; ++i) {
                    /// 衝突点xから半球上のランダムな方向にレイを飛ばす
                    Ray _ray;
                    RayHit _hit;
                    diffuseSample(hit.point, hit.normal, _ray);

                    /// もしBodyに当たったら,その発光量を加算する
                    if (hitScene(_ray, _hit)) {
                        reflectRadiance += bodies[hit.idx].getKd().cwiseProduct(bodies[_hit.idx].getEmission());
                    }
                }
                /// 自己発光 + 反射光
                image.pixels[p_idx] = bodies[hit.idx].getEmission() + reflectRadiance / static_cast<double>(samples);
            } else {
                image.pixels[p_idx] = bgColor;
            }

        }
    }

    return image;
}

Image Renderer::_directIlluminationRender(const unsigned int &samples) const {
    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());
    /// フィルム上のピクセル全てに向けてレイを飛ばす
#pragma omp parallel for
    for (int p_y = 0; p_y < image.height; p_y++) {
        for (int p_x = 0; p_x < image.width; p_x++) {
            const int p_idx = p_y * image.width + p_x;
            Ray ray;
            RayHit hit;
            camera.filmView(p_x, p_y, ray);

            if (hitScene(ray, hit)) {
                if (bodies[hit.idx].isLight()) {    // 光源ならそのemissionを加える

                    image.pixels[p_idx] = bodies[hit.idx].getEmission();

                } else {

                    Color reflectRadiance = Color::Zero();
                    Material material = bodies[hit.idx].material;

                    if (bodies[hit.idx].type == Body::Type::Sphere){    // wall(Sphere)ならdiffuseSample

                        for (int i = 0; i < samples; ++i) {
                            Ray _ray;
                            RayHit _hit;

                            diffuseSample(hit.point, hit.normal, _ray);

                            /// もしBodyに当たったら,その発光量を加算する
                            if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
                                reflectRadiance += bodies[hit.idx].getKd().cwiseProduct(bodies[_hit.idx].getEmission());
                            }
                        }

                    } else if (bodies[hit.idx].type == Body::Type::Cylinder) {  // hair(Cylinder)ならmodel分別

                            if (material.shadingModel == Material::KAJIYA_KAY) {    // Kajiya Kay

                                for (int i = 0; i < samples; ++i) {
                                    Ray _ray;
                                    RayHit _hit;

                                    diffuseSampleHair(ray, hit.point, bodies[hit.idx].cylinder.axis, _ray);

                                    if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
                                        // 光源からの放射輝度
                                        Color emission = bodies[_hit.idx].getEmission();

                                        // 髪の方向
                                        Eigen::Vector3d H = bodies[hit.idx].cylinder.axis.normalized();

                                        // 光源方向 L
                                        Eigen::Vector3d L = (_hit.point - hit.point).normalized();

                                        // 視線方向 V
                                        Eigen::Vector3d V = -ray.dir.normalized();

                                        // 法線ベクトル N（Kajiya-Kayモデル）
                                        Eigen::Vector3d N = V - H * (V.dot(H));
                                        double N_norm = N.norm();
                                        if (N_norm > 1e-6) {
                                            N /= N_norm;
                                        } else {
                                            N = H.unitOrthogonal();
                                        }

                                        // H.dot(L) のクランプ
                                        double hDotL = std::clamp(H.dot(L), -1.0, 1.0);
                                        double diffuseTerm = sqrt(std::max(0.0, 1.0 - hDotL * hDotL));
                                        //double diffuseTerm = std::max(0.0, 1.0 - hDotL * hDotL) / sqrt(std::max(0.0, 1.0 - hDotL * hDotL));

                                        // H.dot(V) のクランプ
                                        double hDotV = std::clamp(H.dot(V), -1.0, 1.0);
                                        //double specularTerm = pow(sqrt(std::max(0.0, 1.0 - hDotV * hDotV)), bodies[hit.idx].material.n);
                                        double specularTerm = pow(sqrt(std::max(0.0, 1.0 - hDotL * hDotL)) * sqrt(std::max(0.0, 1.0 - hDotV * hDotV)) - hDotL * hDotV,
                                                                  bodies[hit.idx].material.n);

                                        // マテリアル特性の取得
                                        Color kd = bodies[hit.idx].getKd();
                                        double ks = bodies[hit.idx].material.ks;

                                        // Kajiya Kayモデルを計算
                                        Color shading = kd * diffuseTerm + ks * specularTerm * Color::Ones();

                                        reflectRadiance += shading.cwiseProduct(emission);
                                    }
                                }

                            } else if (material.shadingModel == Material::MARSCHNER) {  // Marschner

                                for (int i = 0; i < samples; ++i) {
                                    Ray _ray;
                                    RayHit _hit;

                                    marschnerSample(ray, hit.point, bodies[hit.idx].cylinder.axis, material, _ray);

                                    if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
                                        // 光源からの放射輝度
                                        Color emission = bodies[_hit.idx].getEmission();

                                        // 髪の方向 H
                                        Eigen::Vector3d H = bodies[hit.idx].cylinder.axis.normalized();

                                        // 光源方向 L
                                        Eigen::Vector3d L = (_hit.point - hit.point).normalized();

                                        // 視線方向 V
                                        Eigen::Vector3d V = -ray.dir.normalized();

                                        // Marschnerモデルを計算
                                        Color shading = marschnerShading(V, L, H, material);

                                        reflectRadiance += shading.cwiseProduct(emission);
                                    }
                                }
                            }

                    }
                    /// 自己発光 + 反射光
                    image.pixels[p_idx] = reflectRadiance / static_cast<double>(samples);
                }
            } else {
                image.pixels[p_idx] = bgColor;
            }
        }
    }

    return image;
}


void Renderer::diffuseSample(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray) const {
    /// normalの方向をy軸とした正規直交基底を作る (u, normal, v)
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    const double phi = 2.0 * EIGEN_PI * rand();
    const double theta = asin(sqrt(rand()));

    /// theta, phiから出射ベクトルを計算
    const double _x = sin(theta) * cos(phi);
    const double _y = cos(theta);
    const double _z = sin(theta) * sin(phi);

    /// ローカルベクトルからグローバルのベクトルに変換
    out_Ray.dir = _x * u + _y * normal + _z * v;


    out_Ray.org = incidentPoint;
}

//void Renderer::diffuseSampleHair(Ray &in_ray, const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &axis, Ray &out_Ray) const {
//    /// axisの方向をy軸とした正規直交基底を作る (w, axis, v)
//    Eigen::Vector3d w, v;
//    Eigen::Vector3d axis_norm = axis.normalized();
//    computeLocalFrameHair(axis_norm, w, v);
//
//    /// theta_iを計算
//    Eigen::Vector3d omega_i = -in_ray.dir.normalized(); // in_rayの逆ベクトル
//    const double theta_i = acos(omega_i.dot(w));
//
//    /// phi_iを計算
//    Eigen::Vector3d omega_i_proj = omega_i - (omega_i.dot(w) )* w;  // omega_iをw-v平面に投影
//    omega_i_proj.normalize();
//    const double phi_i = atan2(omega_i_proj.dot(w.cross(v)), omega_i_proj.dot(v));
//
//
//    /// 拡散反射
//    const double theta_r = -theta_i;
//    const double phi_r = -phi_i;
//
//    /// theta, phiから出射ベクトルを計算
//    const double _x = sin(theta_r) * cos(phi_r);
//    const double _y = cos(theta_r);
//    const double _z = sin(theta_r) * sin(phi_r);
//
//    /// ローカルベクトルからグローバルのベクトルに変換
//    out_Ray.dir = _x * w + _y * axis + _z * v;
//
//
//    out_Ray.org = incidentPoint;
//}

void Renderer::diffuseSampleHair(Ray &in_ray, const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &hairDir,
                                 Ray &out_Ray) const {
    // 視線方向を計算
    Eigen::Vector3d V = -in_ray.dir.normalized();

    // 髪の方向ベクトル H を正規化
    Eigen::Vector3d H = hairDir.normalized();

    // 法線ベクトル N（Kajiya-Kayモデル）
    Eigen::Vector3d N = V - H * (V.dot(H));
    double N_norm = N.norm();
    if (N_norm > 1e-6) {
        N /= N_norm;
    } else {
        N = H.unitOrthogonal();
    }

    // ランダムな半球サンプリング
    Eigen::Vector3d u, v;
    computeLocalFrame(N, u, v);
    double phi = 2.0 * EIGEN_PI * rand();
    double cosTheta = rand();
    double sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // ローカル座標でのランダムな方向
    Eigen::Vector3d localDir = sinTheta * cos(phi) * u + sinTheta * sin(phi) * v + cosTheta * N;

    // グローバル座標に変換
    out_Ray.dir = localDir.normalized();
    out_Ray.org = incidentPoint;
}


void Renderer::computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v) {
    if (fabs(w.x()) > 1e-3)
        u = Eigen::Vector3d::UnitY().cross(w).normalized();
    else
        u = Eigen::Vector3d::UnitX().cross(w).normalized();

    v = w.cross(u);
}

//void Renderer::computeLocalFrameHair(const Eigen::Vector3d &u, Eigen::Vector3d &w, Eigen::Vector3d &v) {
//    if (fabs(u.x()) > 1e-3)
//        w = Eigen::Vector3d::UnitY().cross(u).normalized();
//    else
//        w = Eigen::Vector3d::UnitX().cross(u).normalized();
//
//    v = u.cross(w);
//}

Color Renderer::marschnerShading(const Eigen::Vector3d &V, const Eigen::Vector3d &L,
                                 const Eigen::Vector3d &H, const Material &material) const {
    // 定数
    const double PI = EIGEN_PI;

    // 入射角と出射角の計算
    double cosTheta_i = std::clamp(H.dot(V), -1.0, 1.0);
    double cosTheta_r = std::clamp(H.dot(L), -1.0, 1.0);
    double theta_i = acos(cosTheta_i);
    double theta_r = acos(cosTheta_r);

    // 方位角差 Δφ の計算
    // 髪の方向ベクトル H を基準にローカル座標系を構築
    Eigen::Vector3d T = H;
    Eigen::Vector3d B = T.unitOrthogonal(); // Tと直交する単位ベクトル
    Eigen::Vector3d N = T.cross(B);

    double phi_i = atan2(V.dot(B), V.dot(N));
    double phi_r = atan2(L.dot(B), L.dot(N));
    double deltaPhi = phi_r - phi_i;

    // 縦方向の散乱計算
    double M_R = computeLongitudinalScattering(theta_i, theta_r, material.beta_m);

    // 横方向の散乱計算
    double M_phi = computeAzimuthalScattering(deltaPhi, material.beta_n);

    // 吸収の計算（色と吸収係数に基づく）
    Color T_val = (material.sigma_a.array() * material.color.array()).exp();

    // 最終的なシェーディング結果
    Color C = material.kd * M_R * M_phi * T_val;

    return C;
}

void Renderer::marschnerSample(const Ray &in_ray, const Eigen::Vector3d &incidentPoint,
                               const Eigen::Vector3d &hairDir, const Material &material,
                               Ray &out_Ray) const {
    // 髪の方向ベクトル H を正規化
    Eigen::Vector3d H = hairDir.normalized();

    // 視線方向 V を計算（カメラから髪の毛へのベクトル）
    Eigen::Vector3d V = -in_ray.dir.normalized();

    // 入射角 θ_i を計算
    double cosTheta_i = std::clamp(H.dot(V), -1.0, 1.0);
    double theta_i = acos(cosTheta_i);

    // ランダムな数値生成
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);
    double randVal = uniformDist(engine);

    // 散乱モードの選択（R, TT, TRT）
    enum ScatteringMode {
        R, TT, TRT
    };
    ScatteringMode mode;
    // Marschnerモデルでは各モードの寄与率が異なるため、寄与率に基づいて選択
    // 例: R: 60%, TT: 20%, TRT: 20%
    if (randVal < 0.6) {
        mode = R;
    } else if (randVal < 0.8) {
        mode = TT;
    } else {
        mode = TRT;
    }

    double theta_r = 0.0;
    double deltaPhi = 0.0;
    const double PI = EIGEN_PI;

    switch (mode) {
        case R:
            // 一次反射: 出射角 θ_r は入射角 θ_i に基づく
            // θ_r = θ_i（鏡面反射）
            theta_r = theta_i;
            // 方位角差 Δφ は0に近い
            deltaPhi = 0.0;
            break;
        case TT:
            // 透過反射: 出射角 θ_r は入射角 θ_i に基づく
            // θ_r = θ_i（対向方向への透過）
            theta_r = theta_i;
            // 方位角差 Δφ はπに近い
            deltaPhi = PI;
            break;
        case TRT:
            // 内部反射: 出射角 θ_r は入射角 θ_i に基づく
            // θ_r = θ_i（同一側への内部反射）
            theta_r = theta_i;
            // 方位角差 Δφ は2πに近い
            deltaPhi = 2.0 * PI;
            break;
    }

    // 追加のランダム性を加える（髪の毛のカールや広がりを表現）
    double sigma_m = material.beta_m;
    double sigma_n = material.beta_n;
    std::normal_distribution<double> thetaDist(theta_r, sigma_m);
    std::normal_distribution<double> phiDist(deltaPhi, sigma_n);

    // サンプリングされたθ_rとΔφを取得
    double sampledTheta_r = thetaDist(engine);
    double sampledDeltaPhi = phiDist(engine);

    // サンプリング値をクランプ
    sampledTheta_r = std::clamp(sampledTheta_r, 0.0, PI);
    sampledDeltaPhi = std::fmod(sampledDeltaPhi, 2.0 * PI);

    // ローカル座標系の構築（HをZ軸とする）
    Eigen::Vector3d Z = H;
    Eigen::Vector3d X = (Eigen::Vector3d::UnitY().cross(Z)).normalized();
    Eigen::Vector3d Y = Z.cross(X).normalized();

    // 出射方向の計算
    Eigen::Vector3d L = sin(sampledTheta_r) * cos(sampledDeltaPhi) * X +
                        sin(sampledTheta_r) * sin(sampledDeltaPhi) * Y +
                        cos(sampledTheta_r) * Z;
    L.normalize();

    // 出射レイの設定
    out_Ray.org = incidentPoint;
    out_Ray.dir = L;
}

double Renderer::computeLongitudinalScattering(double theta_i, double theta_r, double beta_m) const {
    // 縦方向の散乱項（ガウス分布を使用）
    double deltaTheta = theta_r - theta_i;
    double sigma = beta_m;
    double result = exp(-(deltaTheta * deltaTheta) / (2 * sigma * sigma)) / (sqrt(2 * EIGEN_PI) * sigma);
    return result;
}

double Renderer::computeAzimuthalScattering(double deltaPhi, double beta_n) const {
    // 横方向の散乱項（ガウス分布を使用）
    double sigma = beta_n;
    double result = exp(-(deltaPhi * deltaPhi) / (2 * sigma * sigma)) / (sqrt(2 * EIGEN_PI) * sigma);
    return result;
}