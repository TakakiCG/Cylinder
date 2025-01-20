//
// Created by kango on 2023/04/03.
//

#include "Renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>

const double PI = EIGEN_PI;
int trace_num = 0;
#define M_PI 3.14159265358979323846


/// シーン内の物体、カメラ、背景色を初期化する
Renderer::Renderer(const std::vector<Body> &bodies, Camera camera, Color bgColor)
        : bodies(bodies), camera(std::move(camera)), bgColor(std::move(bgColor)), engine(0), dist(0, 1){
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

//Image Renderer::directIlluminationRender(const unsigned int &samples) const {
//    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());
//    /// フィルム上のピクセル全てに向けてレイを飛ばす
//#pragma omp parallel for
//    for (int p_y = 0; p_y < image.height; p_y++) {
//        for (int p_x = 0; p_x < image.width; p_x++) {
//            const int p_idx = p_y * image.width + p_x;
//            Ray ray;
//            RayHit hit;
//            camera.filmView(p_x, p_y, ray);
//
//            if (hitScene(ray, hit)) {
//                Color reflectRadiance = Color::Zero();
//                for (int i = 0; i < samples; ++i) {
//                    /// 衝突点xから半球上のランダムな方向にレイを飛ばす
//                    Ray _ray;
//                    RayHit _hit;
//                    diffuseSample(hit.point, hit.normal, _ray);
//
//                    /// もしBodyに当たったら,その発光量を加算する
//                    if (hitScene(_ray, _hit)) {
//                        reflectRadiance += bodies[hit.idx].getKd().cwiseProduct(bodies[_hit.idx].getEmission());
//                    }
//                }
//                /// 自己発光 + 反射光
//                image.pixels[p_idx] = bodies[hit.idx].getEmission() + reflectRadiance / static_cast<double>(samples);
//            } else {
//                image.pixels[p_idx] = bgColor;
//            }
//
//        }
//    }
//
//    return image;
//}

/// Kajiya Kay
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
                                Ray _ray; RayHit _hit;

                                const double kd = bodies[hit.idx].getKd().maxCoeff();
                                const double ks = bodies[hit.idx].getKs().maxCoeff();
                                const double r = rand();

                                ///
                                if(r < kd){
                                    //diffuseの処理
                                    diffuseSampleHair(hit.point, hit.normal, _ray);

                                    if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {

                                        // 光源からの放射輝度
                                        Color emission = bodies[_hit.idx].getEmission();

//                                        // 髪の方向
//                                        const Eigen::Vector3d H = bodies[hit.idx].cylinder.axis.normalized();
//
//                                        // 光源方向 L
//                                        const Eigen::Vector3d L = (_hit.point - hit.point).normalized();
//
//                                        // 視線方向 V
//                                        const Eigen::Vector3d V = -ray.dir.normalized();
//
//
//                                        const double hDotL = std::clamp(H.dot(L), -1.0, 1.0);
//                                        // 拡散BRDF
//                                        const double diffuseTerm = sqrt(std::max(0.0, 1.0 - hDotL * hDotL));

                                        //reflectRadiance += diffuseTerm * bodies[hit.idx].getKd().cwiseProduct(emission) / kd;
                                        reflectRadiance += bodies[hit.idx].getKd().cwiseProduct(emission) / kd;

                                    }
                                }
                                else if (r < kd + ks){
                                    //specularの処理
                                    specularSampleHair(hit.point, hit.normal, _ray, bodies[hit.idx].material.n);
                                    //diffuseSample(hit.point, hit.normal, _ray);

                                    if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
                                        // 光源からの放射輝度
                                        Color emission = bodies[_hit.idx].getEmission();

//                                        // 髪の方向
//                                        const Eigen::Vector3d H = bodies[hit.idx].cylinder.axis.normalized();
//
//                                        // 光源方向 L
//                                        const Eigen::Vector3d L = (_hit.point - hit.point).normalized();
//
//                                        // 視線方向 V
//                                        const Eigen::Vector3d V = -ray.dir.normalized();
//
//
//                                        const double hDotL = std::clamp(H.dot(L), -1.0, 1.0);
//
//                                        const double hDotV = std::clamp(H.dot(V), -1.0, 1.0);
//
//                                        // 鏡面BRDF
//                                        const double specularTerm = pow(sqrt(std::max(0.0, 1.0 - hDotL * hDotL)) * sqrt(std::max(0.0, 1.0 - hDotV * hDotV)) - hDotL * hDotV,
//                                                                  bodies[hit.idx].material.n);

                                        //reflectRadiance += PI * specularTerm * bodies[hit.idx].getKs().cwiseProduct(emission) / ks;
                                        reflectRadiance += bodies[hit.idx].getKs().cwiseProduct(emission) / ks;
                                    }
                                }

                            }

                        }
//                        else if (material.shadingModel == Material::MARSCHNER) {  // Marschner
//                            for (int i = 0; i < samples; ++i) {
//                                Ray _ray;
//                                RayHit _hit;
//
//                                marschnerSampleHair(ray, hit.point, bodies[hit.idx].cylinder.axis, _ray);
//
//                                if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
//                                    reflectRadiance += bodies[hit.idx].getKd().cwiseProduct(bodies[_hit.idx].getEmission());
//                                }
//                            }
//                        }

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

Image Renderer::passTracingRender(const unsigned int &samples) const {
    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());
    /// フィルム上のピクセル全てに向けてレイを飛ばす
#pragma omp parallel for
    for (int p_y = 0; p_y < image.height; p_y++) {
        for (int p_x = 0; p_x < image.width; p_x++) {
            const int p_idx = p_y * image.width + p_x;
            Color color;
            // ピクセルに飛ばすレイを生成
            Ray ray;
            RayHit hit;
            camera.filmView(p_x, p_y, ray);

            if (hitScene(ray, hit)) {
                Color accumulatedColor = Color::Zero();
                for (int i = 0; i < samples; ++i) {
                    accumulatedColor += trace(ray, hit);
                }
                color = accumulatedColor / static_cast<double>(samples);
            } else {
                color = bgColor;
            }
            image.pixels[p_idx] = color;
        }
    }
    return image;
}

//Image Renderer::_directIlluminationRender_anti_areas(const unsigned int &samples, const unsigned int &areas_n_samples) const {
//    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());
//    /// フィルム上のピクセル全てに向けてレイを飛ばす
//#pragma omp parallel for
//    for (int p_y = 0; p_y < image.height; p_y++) {
//        for (int p_x = 0; p_x < image.width; p_x++) {
//            //const unsigned int areas_n_samples = 10;
//            Color pixel_color = Color::Zero();
//            const int p_idx = p_y * image.width + p_x;
//
//            for (int areas_sample_idx = 0; areas_sample_idx < areas_n_samples; areas_sample_idx++) {
//                Ray ray;
//                RayHit hit;
//                const double r_x = rand();
//                const double r_y = rand();
//                camera.filmView_random(p_x, p_y, r_x, r_y, ray);
//
//                if (hitScene(ray, hit)) {
//                    if (bodies[hit.idx].isLight()) {    // 光源ならそのemissionを加える
//
//                        //image.pixels[p_idx] = bodies[hit.idx].getEmission();
//                        pixel_color += bodies[hit.idx].getEmission();
//
//                    } else {
//
//                        Color reflectRadiance = Color::Zero();
//                        Material material = bodies[hit.idx].material;
//
//                        if (bodies[hit.idx].type == Body::Type::Sphere) {    // wall(Sphere)ならdiffuseSample
//                            for (int i = 0; i < samples; ++i) {
//                                Ray _ray;
//                                RayHit _hit;
//
//                                diffuseSample(hit.point, hit.normal, _ray);
//
//                                /// もしBodyに当たったら,その発光量を加算する
//                                if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
//                                    reflectRadiance += bodies[hit.idx].getKd().cwiseProduct(
//                                            bodies[_hit.idx].getEmission());
//                                }
//                            }
//
//                        } else if (bodies[hit.idx].type == Body::Type::Cylinder) {  // hair(Cylinder)ならmodel分別
//
//                            if (material.shadingModel == Material::KAJIYA_KAY) {    // Kajiya Kay
//                                for (int i = 0; i < samples; ++i) {
//                                    //std::cout << i << "/" << samples << std::endl;
//                                    Ray _ray;
//                                    RayHit _hit;
//
//                                    //diffuseSampleHair(ray, hit.point, bodies[hit.idx].cylinder.axis, _ray);
//
//                                    // eval diffuse
//                                    if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
//                                        // 光源からの放射輝度
//                                        Color emission = bodies[_hit.idx].getEmission();
//
//                                        // 髪の方向
//                                        Eigen::Vector3d H = bodies[hit.idx].cylinder.axis.normalized();
//
//                                        // 光源方向 L
//                                        Eigen::Vector3d L = (_hit.point - hit.point).normalized();
//
//                                        // 視線方向 V
//                                        Eigen::Vector3d V = -ray.dir.normalized();
//
//                                        // 法線ベクトル N（Kajiya-Kayモデル）
//                                        Eigen::Vector3d N = V - H * (V.dot(H));
//                                        double N_norm = N.norm();
//                                        if (N_norm > 1e-6) {
//                                            N /= N_norm;
//                                        } else {
//                                            N = H.unitOrthogonal();
//                                        }
//
//                                        // H.dot(L) のクランプ
//                                        double hDotL = std::clamp(H.dot(L), -1.0, 1.0);
//                                        double diffuseTerm = sqrt(std::max(0.0, 1.0 - hDotL * hDotL));
//                                        //double diffuseTerm = std::max(0.0, 1.0 - hDotL * hDotL) / sqrt(std::max(0.0, 1.0 - hDotL * hDotL));
//
//                                        // H.dot(V) のクランプ
//                                        double hDotV = std::clamp(H.dot(V), -1.0, 1.0);
//                                        //double specularTerm = pow(sqrt(std::max(0.0, 1.0 - hDotV * hDotV)), bodies[hit.idx].material.n);
//                                        double specularTerm = pow(sqrt(std::max(0.0, 1.0 - hDotL * hDotL)) *
//                                                                  sqrt(std::max(0.0, 1.0 - hDotV * hDotV)) -
//                                                                  hDotL * hDotV,
//                                                                  bodies[hit.idx].material.n);
//
//                                        // マテリアル特性の取得
//                                        Color kd = bodies[hit.idx].getKd();
//                                        double ks = bodies[hit.idx].material.ks;
//
//                                        // Kajiya Kayモデルを計算
//                                        Color shading = kd * diffuseTerm + ks * specularTerm * Color::Ones();
//
//                                        reflectRadiance += shading.cwiseProduct(emission);
//
//                                    }
//                                }
//
//                            } else if (material.shadingModel == Material::MARSCHNER) {  // Marschner
//                                for (int i = 0; i < samples; ++i) {
//                                    Ray _ray;
//                                    RayHit _hit;
//
//                                    marschnerSample(ray, hit.point, bodies[hit.idx].cylinder.axis, material, _ray);
//
//                                    if (hitScene(_ray, _hit) && bodies[_hit.idx].isLight()) {
//                                        // 光源からの放射輝度
//                                        Color emission = bodies[_hit.idx].getEmission();
//
//                                        // 髪の方向 H
//                                        Eigen::Vector3d H = bodies[hit.idx].cylinder.axis.normalized();
//
//                                        // 光源方向 L
//                                        Eigen::Vector3d L = (_hit.point - hit.point).normalized();
//
//                                        // 視線方向 V
//                                        Eigen::Vector3d V = -ray.dir.normalized();
//
//                                        // Marschnerモデルを計算
//                                        Color shading = marschnerShading(V, L, H, material);
//
//                                        reflectRadiance += shading.cwiseProduct(emission);
//                                    }
//                                }
//                            }
//
//                        }
//                        /// 自己発光 + 反射光
//                        // image.pixels[p_idx] = reflectRadiance / static_cast<double>(samples);   // サンプル数分足した放射輝度の平均
//                        pixel_color += reflectRadiance / static_cast<double>(samples);
//                    }
//                } else {
//                    pixel_color += bgColor;
//                }
//            }
//            image.pixels[p_idx] = pixel_color / static_cast<double>(areas_n_samples);
//        }
//    }
//
//    return image;
//}

Color Renderer::trace(const Ray &ray, const RayHit &hit) const {
    if(!hit.isHit()) {
        return Color::Zero();
    }

    const auto hitBody = bodies[hit.idx];

    // 衝突物体の自己発光を足す
    Color out_color = hitBody.getEmission();

    // ロシアンルーレット
    const double kd = hitBody.getKd().maxCoeff();
    const double ks = hitBody.getKs().maxCoeff();
    const double r = rand();

    /// added
    // ロシアンルーレット確率(例)
    double p_survive = 0.9;  // 適宜調整
    if (rand() > p_survive) {
        return out_color; // 打ち切り
    }
    double rr_scale = 1.0 / p_survive; // Throughput 補正

    // 入射方向
    Eigen::Vector3d wi = -ray.dir.normalized();
    ///

    if (hitBody.type == Body::Type::Sphere) {
        if(r < kd){
            Ray _ray; RayHit _hit;
            diffuseSample(hit.point, hit.normal, _ray);

            hitScene(_ray, _hit);
            if(bodies[_hit.idx].isLight()) {
                return out_color += hitBody.getKd().cwiseProduct(bodies[_hit.idx].getEmission()) / kd;
            }
            else {
                out_color += hitBody.getKd().cwiseProduct(trace(_ray, _hit));
            }
        }
    }
    else if(hitBody.type == Body::Type::Cylinder && hitBody.material.shadingModel == Material::KAJIYA_KAY) {
        if(r < kd) {
            Ray _ray; RayHit _hit;
            diffuseSampleHair(hit.point, hit.normal, _ray);

            /// Marschner
            //marschnerSampleHair(ray, hit.point, hitBody.cylinder.axis, _ray);

            hitScene(_ray, _hit);

//            // レイを飛ばす髪の軸方向
//            const Eigen::Vector3d T = bodies[hit.idx].cylinder.axis.normalized();
//            // 次のレイの方向 L
//            const Eigen::Vector3d L = (_hit.point - hit.point).normalized();
//            // 前の髪から次の髪の方向(視線方向) V
//            const Eigen::Vector3d V = -ray.dir.normalized();
//
//            const double tDotL = std::clamp(T.dot(L), -1.0, 1.0);
//            // 拡散BRDF
//            const double diffuseTerm = sqrt(std::max(0.0, 1.0 - tDotL * tDotL));

            if(bodies[_hit.idx].isLight()) {
                // 光源の輝度
                const Color emission = bodies[_hit.idx].getEmission();
                //out_color += diffuseTerm * hitBody.getKd().cwiseProduct(emission) / kd;
                out_color += hitBody.getKd().cwiseProduct(emission) / kd;   // 重点的サンプリング済(ニュートン法)
            }
            else{
                //out_color += diffuseTerm * hitBody.getKd().cwiseProduct(trace(_ray, _hit)) / kd;
                out_color += hitBody.getKd().cwiseProduct(trace(_ray, _hit)) / kd;
            }
        }
        else if(r < kd + ks) {
            Ray _ray; RayHit _hit;
            specularSampleHair(hit.point, hit.normal, _ray, hitBody.material.n);

            /// Marschner
            //marschnerSampleHair(ray, hit.point, hitBody.cylinder.axis, _ray);

            hitScene(_ray, _hit);

//            // レイを飛ばす髪の軸方向
//            const Eigen::Vector3d T = hitBody.cylinder.axis.normalized();
//            // 次のレイの方向 L
//            const Eigen::Vector3d L = (_hit.point - hit.point).normalized();
//            // 次の衝突点からレイを飛ばした髪の方向(視線方向) V
//            const Eigen::Vector3d V = -ray.dir.normalized();
//
//            const double tDotL = std::clamp(T.dot(L), -1.0, 1.0);
//            const double tDotV = std::clamp(T.dot(V), -1.0, 1.0);
//            // 鏡面BRDF
//            const double specularTerm = pow(sqrt(std::max(0.0, 1.0 - tDotL * tDotL)) * sqrt(std::max(0.0, 1.0 - tDotV * tDotV)) - tDotL * tDotV,
//                                      hitBody.material.n);

            if(bodies[_hit.idx].isLight()) {
                // 光源の輝度
                const Color emission = bodies[_hit.idx].getEmission();
                //out_color += specularTerm * hitBody.getKs().cwiseProduct(emission) / ks;
                out_color += hitBody.getKs().cwiseProduct(emission) / ks;
            }
            else {
                //out_color+= specularTerm * hitBody.getKs().cwiseProduct(trace(_ray, _hit));
                out_color += hitBody.getKs().cwiseProduct(trace(_ray, _hit)) / ks;
            }
        }
    }

    return out_color;
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

void Renderer::marschnerSampleHair(const Ray &in_Ray, const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &axis, Ray &out_Ray) {
    /// axis(u)の方向をy軸とした正規直交基底を作る (w, u, v)
    Eigen::Vector3d w, v;
    Eigen::Vector3d u = axis;
    computeLocalFrame(u, w, v);

    // 入射光ベクトルをローカル基底に変換（w, v平面への投影を計算する）
    Eigen::Vector3d neg_in_dir = -in_Ray.dir;

    // w, v平面への投影ベクトルを計算する
    Eigen::Vector3d projected_on_wv = neg_in_dir.dot(w) * w + neg_in_dir.dot(v) * v;

    // 2.1. 入射角theta_iの計算（w, v平面への垂直成分を基に）
    double theta_i = acos(neg_in_dir.dot(u)); // w軸に対する傾き

    // 2.2. 入射角phi_iの計算（w, v平面上での投影方向）
    double phi_i = atan2(projected_on_wv.dot(v), projected_on_wv.dot(w)); // w, v平面での角度

    // 2.3. theta_iを-π/2 ~ +π/2の範囲に調整
    if (neg_in_dir.dot(w) < 0) {
        // -in_Rayがw, v平面で-u方向にある場合、theta_iは負になるべき
        theta_i = -theta_i;
    }


    /// MとNによる出射角の決定
    /// ・・・
    /// 今は鏡面反射
    const double theta_r = - theta_i;
    const double phi_r = phi_i;

    /// theta, phiから出射ベクトルを計算
    const double _x = cos(theta_r) * sin(phi_r);
    const double _y = sin(theta_r);
    const double _z = cos(theta_r) * cos(phi_r);

    /// ローカルベクトルからグローバルのベクトルに変換
    out_Ray.dir = _x * w + _y * u + _z * v;

    out_Ray.org = incidentPoint;
}


//void Renderer::diffuseSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal,
//                                 Ray &out_Ray) const {
//    /// normalの方向をy軸とした正規直交基底を作る (u, normal, v)
//    Eigen::Vector3d u, v;
//    computeLocalFrame(normal, u, v);
//
//    /// BRDFそのまま用いる
//    const double phi = 2 * PI * rand();
//    const double theta = acos(1 - rand());
//
//    /// theta, phiから出射ベクトルを計算
//    const double _x = sin(theta) * cos(phi);
//    const double _y = cos(theta);
//    const double _z = sin(theta) * sin(phi);
//
//    /// ローカルベクトルからグローバルのベクトルに変換
//    out_Ray.dir = _x * u + _y * normal + _z * v;
//
//    out_Ray.org = incidentPoint;
//}

void Renderer::diffuseSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal,
                                 Ray &out_Ray) const {
    /// normalの方向をy軸とした正規直交基底を作る (u, normal, v)
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    /// ニュートン法でthetaの値を求める
    //const double y = rand();

    try {
        const double theta = newton_method();
        const double phi = 2 * PI * rand();
        //const double theta = acos(1 - rand());

        /// theta, phiから出射ベクトルを計算
        const double _x = sin(theta) * cos(phi);
        const double _y = cos(theta);
        const double _z = sin(theta) * sin(phi);

        /// ローカルベクトルからグローバルのベクトルに変換
        out_Ray.dir = _x * u + _y * normal + _z * v;

        out_Ray.org = incidentPoint;

    } catch (const std::exception &e) {
        std::cerr << "エラー: " << e.what() << std::endl;
    }
}

double Renderer::newton_method() const {
    const double y = rand();

    /// 初期値
    double x = 0.5 * PI;

    for (int i = 0; i < 100; ++i) {
        double fx = (x - 0.5 * std::sin(2.0 * x)) / PI - y;
        double fpx = (1.0 - std::cos(2.0 * x)) / PI;

        if (std::fabs(fpx) < 1e-20) {
            throw std::runtime_error("Derivative too small, no convergence.");
        }

        double x_new = x - fx / fpx;

        if (std::fabs(x_new - x) < 1e-12) { // xの変化量が誤差の範囲内になったら終了
            return x_new;
        }

        x = x_new;
    }
    throw std::runtime_error("Newton method did not converge within the given max_iter.");
}

void Renderer::specularSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, const double & n) const {
    /// normalの方向をy軸とした正規直交基底を作る (u, normal, v)
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    double theta;
    double phi;

//    while(true){
        phi = 2.0 * PI * rand();
        /// cos^nの逆関数法によるサンプリング
        const double r = rand();
        //theta = acos(pow(r, 1.0 / (n + 1.0)));
        theta = acos(pow(1.0 - r, 1.0 / (n + 1.0)));

//        /// sinを棄却法でサンプリング
//        const double sinTheta = sin(theta);
//        if(rand() <= sinTheta) {
//            break;
//        }
//    }
//
//    const double phi = 2.0 * EIGEN_PI * rand();
//    const double theta = acos(std::pow(rand(), 1 / n + 1));

    /// theta, phiから出射ベクトルを計算
    const double _x = sin(theta) * cos(phi);
    const double _y = cos(theta);
    const double _z = sin(theta) * sin(phi);

    /// ローカルベクトルからグローバルのベクトルに変換
    out_Ray.dir = _x * u + _y * normal + _z * v;
    out_Ray.org = incidentPoint;
}

void Renderer::computeLocalFrame(const Eigen::Vector3d &w, Eigen::Vector3d &u, Eigen::Vector3d &v) {
    if (fabs(w.x()) > 1e-3)
        u = Eigen::Vector3d::UnitY().cross(w).normalized();
    else
        u = Eigen::Vector3d::UnitX().cross(w).normalized();

    v = w.cross(u);
}

/// Marschner
//Color Renderer::fur_bsdf(const Eigen::Vector3d &wi, const Eigen::Vector3d &wo, const BSDFParams &params) const {
//    const double eta = params.eta;
//    const double alpha_R = params.alpha_R;
//    const double alpha_TT = - alpha_R / 2;
//    const double alpha_TRT = - 3 * alpha_R / 2;
//    const double beta_R = params.beta_R;
//    const double beta_TT = beta_R / 2;
//    const double beta_TRT = 2 * beta_R;
//
//    /// wo:視線方向　wi:ライト方向
//    const double sinTheta_o = wo.x();
//    const double cosTheta_o = 1.0 - sqrt(sinTheta_o);
//    const double phi_o = atan2(wo.z(), wo.y());
//
//    const double sinTheta_i = wi.x();
//    const double cosTheta_i = 1.0 - sqrt(sinTheta_i);
//    const double phi_i = atan2(wi.z(), wi.y());
//
//    const double sinTheta_t = sinTheta_o / eta;
//
//}