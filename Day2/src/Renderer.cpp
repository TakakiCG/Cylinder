//
// Created by kango on 2023/04/03.
//

#include "Renderer.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <fstream>

const double PI = EIGEN_PI;
int trace_num = 0;
#define M_PI 3.14159265358979323846

/// シーン内の物体、カメラ、背景色を初期化する
Renderer::Renderer(const std::vector<Body> &bodies, Camera camera, Color bgColor)
        : bodies(bodies), camera(std::move(camera)), bgColor(std::move(bgColor)), engine(0), dist(0, 1), exceptionCount(0){
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

    /// Visualize theta and phi
    const int TARGET_PX = camera.getFilm().resolution.x() / 2;
    const int TARGET_PY = camera.getFilm().resolution.y() / 3;

    exceptionCount = 0; // カウントの初期化


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
                    //accumulatedColor += trace(ray, hit);
                    //accumulatedColor += traceMarschner(ray, hit);

                    /// Visualize theta and phi
                    bool record = (p_x == TARGET_PX && p_y == TARGET_PY);
                    accumulatedColor += trace(ray, hit, record, i); // 第4引数＝サンプル番号
                }
                color = accumulatedColor / static_cast<double>(samples);
            } else {
                color = bgColor;
            }
            image.pixels[p_idx] = color;
        }
    }

    {
        std::ofstream csv("ray_angles2.csv");
        csv << "sampleIdx,theta_in,phi_in,theta_out,phi_out,hitType,scatterType,reason\n";
        for (const auto& s : g_samples) {
            csv << s.sampleIdx      << ','
                << s.incidentTheta  << ','
                << s.incidentPhi  << ','
                << s.outgoingTheta  << ','
                << s.outgoingPhi    << ','
                << s.hitType        << ','
                << s.scatterType    << ','
                << s.reason         << '\n';
        }
        std::cout << "Saved " << g_samples.size()
                  << " rows to ray_angles.csv\n";
    }

    printExceptionCount(); // カウントを出力
    return image;
}

Image Renderer::passTracingRender_antiAreas(const unsigned int &samples) const {
    Image image(camera.getFilm().resolution.x(), camera.getFilm().resolution.y());

    exceptionCount = 0; // カウントの初期化

    const unsigned int subPixelSamples = 8; // 各ピクセルをサブピクセルに分割

    /// フィルム上のピクセル全てに向けてレイを飛ばす
#pragma omp parallel for
    for (int p_y = 0; p_y < image.height; p_y++) {
        for (int p_x = 0; p_x < image.width; p_x++) {
            const int p_idx = p_y * image.width + p_x;
            Color accumulatedColor = Color::Zero();

            for (unsigned int s = 0; s < subPixelSamples; ++s) {
                for (unsigned int t = 0; t < subPixelSamples; ++t) {
                    // サブピクセル内のランダムな位置を計算
                    double subPixelOffsetX = (s + rand()) / subPixelSamples;
                    double subPixelOffsetY = (t + rand()) / subPixelSamples;

                    // レイを生成
                    Ray ray;
                    camera.filmView(p_x + subPixelOffsetX, p_y + subPixelOffsetY, ray);

                    RayHit hit;
                    if (hitScene(ray, hit)) {
                        Color subPixelColor = Color::Zero();
                        for (unsigned int i = 0; i < samples; ++i) {
                            subPixelColor += trace(ray, hit);
                        }
                        subPixelColor /= static_cast<double>(samples);
                        accumulatedColor += subPixelColor;
                    } else {
                        accumulatedColor += bgColor;
                    }
                }
            }

            // サブピクセルの平均を計算してピクセルに格納
            image.pixels[p_idx] = accumulatedColor / static_cast<double>(subPixelSamples * subPixelSamples);
        }
    }

    printExceptionCount(); // カウントを出力
    return image;
}

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
            //diffuseSampleHair(hit.point, hit.normal, _ray);

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
                //out_color += diffuseTerm * hitBody.getKd().cwiseProduct(emission) / kd; // sin１個のみに沿わせる
                out_color += hitBody.getKd().cwiseProduct(emission) / kd;   // 重点的サンプリング済(ニュートン法)
            }
            else{
                //out_color += diffuseTerm * hitBody.getKd().cwiseProduct(trace(_ray, _hit)) / kd;
                out_color += hitBody.getKd().cwiseProduct(trace(_ray, _hit)) / kd;
            }
        }
        else if(r < kd + ks) {
            Ray _ray; RayHit _hit;
            //specularSampleHair(hit.point, hit.normal, _ray, hitBody.material.n);

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

Color Renderer::trace(const Ray &ray, const RayHit &hit, bool recordAngles, int sampleIdx) const {
    if(!hit.isHit()) {
        return Color::Zero();
    }

    const auto hitBody = bodies[hit.idx];

    // 衝突物体の自己発光を足す
    Color out_color = hitBody.getEmission();

    /// Visualize theta and phi
    RaySampleData tmp;                 // ← スタック上に一時確保
    tmp.sampleIdx = sampleIdx;
    tmp.hitType   = (hitBody.type == Body::Type::Sphere) ? "Sphere" : "Cylinder";

    if (recordAngles) {
        worldToLocalAngles(hit.normal, -ray.dir.normalized(),
                           tmp.incidentTheta, tmp.incidentPhi);
    }


    // ロシアンルーレット
    const double kd = hitBody.getKd().maxCoeff();
    const double ks = hitBody.getKs().maxCoeff();
//    const double kd = hitBody.material.kd;
//    const double ks = hitBody.material.ks;
    const double r = rand();

    /// added
    // ロシアンルーレット確率(例)
    double p_survive = 0.9;  // 適宜調整
//    if (rand() > p_survive) {
//        return out_color; // 打ち切り
//    }
    if (rand() > p_survive) {
        if (recordAngles) {
            tmp.reason = "RR_terminate";
            std::scoped_lock lk(g_mutex);
            g_samples.emplace_back(tmp);
        }
        return out_color;
    }
    ///

    if (hitBody.type == Body::Type::Sphere) {
        if(r < kd){
            Ray _ray; RayHit _hit;
            //diffuseSample(hit.point, hit.normal, _ray);
            tmp.scatterType = "Diffuse";                 // ← 追加
            diffuseSample(hit.point, hit.normal, _ray, recordAngles, &tmp);

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
            diffuseSampleHair(hit.point, hit.normal, _ray, recordAngles, &tmp);
            tmp.scatterType = "Diffuse";

            /// Marschner
            //marschnerSampleHair(ray, hit.point, hitBody.cylinder.axis, _ray, recordAngles, &tmp);

            hitScene(_ray, _hit);

            if(bodies[_hit.idx].isLight()) {
                // 光源の輝度
                const Color emission = bodies[_hit.idx].getEmission();
                out_color += hitBody.getKd().cwiseProduct(emission) / kd;   // 重点的サンプリング済(ニュートン法)
            }
            else{
                out_color += hitBody.getKd().cwiseProduct(trace(_ray, _hit)) / kd;
            }
        }
        else if(r < kd + ks) {
            Ray _ray; RayHit _hit;
            specularSampleHair(hit.point, hit.normal, _ray, hitBody.material.n, recordAngles, &tmp);
            tmp.scatterType = "Specular";

            /// Marschner
            //marschnerSampleHair(ray, hit.point, hitBody.cylinder.axis, _ray, recordAngles, &tmp);

            hitScene(_ray, _hit);


            if(bodies[_hit.idx].isLight()) {
                // 光源の輝度
                const Color emission = bodies[_hit.idx].getEmission();
                out_color += hitBody.getKs().cwiseProduct(emission) / ks;
            }
            else {
                out_color += hitBody.getKs().cwiseProduct(trace(_ray, _hit)) / ks;
            }
        }
        else{
            tmp.reason = "ScatterSkip";
        }
    }

    if (recordAngles) {
        std::scoped_lock lk(g_mutex);   // OpenMP 排他
        g_samples.emplace_back(tmp);
    }

    return out_color;
}

Color Renderer::traceMarschner(const Ray &ray, const RayHit &hit) const {
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
            /// Marschner
            //marschnerSampleHair(ray, hit.point, hitBody.cylinder.axis, _ray);

            Eigen::Vector3d w, v;
            Eigen::Vector3d u = hitBody.cylinder.axis;
            computeLocalFrame(u, w, v);

            Eigen::Vector3d wi = -ray.dir.normalized();
            // 入射光ベクトルをローカル基底に変換（w, v平面への投影を計算する）
            // w, v平面への投影ベクトルを計算する
            Eigen::Vector3d projected_on_wv = (wi.dot(w) * w + wi.dot(v) * v).normalized();

            // 2.1. 入射角thetaIの計算（w, v平面への垂直成分を基に）
            //double thetaI = acos(neg_in_dir.dot(u)); // w軸に対する傾き
            double thetaI = acos(wi.dot(projected_on_wv));
            if(u.dot(wi) < 0.0){
                thetaI = - thetaI;    // rayのu成分が負ならtheta_iは負
            }

            // 2.2. 入射角phi_oの計算（w, v平面上での投影方向）
            double phiI = atan2(projected_on_wv.dot(v), projected_on_wv.dot(w)); // w, v平面での角度
            if(w.dot(projected_on_wv) < 0.0){
                phiI = - phiI;    // rayのw成分が負ならphi_iは負
            }

            double sinTheta_i = sin(thetaI);
            double cosTheta_i = sqrt(1 - sqrt(sinTheta_i));

            Eigen::Vector3d n = hit.normal.normalized(); //衝突点の法線n
            // 衝突点の法線 と rayをvw平面に投影したベクトル(projected_on_wv) との内積を取る
            double cos_gammaI = n.dot(projected_on_wv);
            double gammaI = std::acos(cos_gammaI);
            double sin_gammaI = std::sqrt(std::max(0.0, 1.0 - cos_gammaI * cos_gammaI));
            // projected_on_wvがv軸より上か下かでhの符号決定(?)
            double sign_h = ((projected_on_wv).dot(v) >= 0.0) ? 1.0 : -1.0;
            double h = sign_h * sin_gammaI;

            // 理想的な鏡面反射と仮定してthetaDとA0_specは出すらしい
            /// R成分のthetaRは鏡面反射とする
//            double wi_dot_n = n.dot(neg_in_dir);
//            const Eigen::Vector3d specular_direction = (2.0 * n * wi_dot_n - neg_in_dir).normalized();
            double thetaR = - thetaI;
            double thetaD = (thetaR - thetaI) / 2.0;
            double eta = hitBody.material.eta;
            const double etaP = std::sqrt(eta * eta - std::sin(thetaD)) / std::cos(thetaD); // d'Eon
            const double gammaT = std::asin(h / etaP);

            /// 続きここから
            double fresnel = FresnelFunction(cos_gammaI, eta);
            



//            Ray _ray; RayHit _hit;
//            hitScene(_ray, _hit);
//            if(bodies[_hit.idx].isLight()) {
//                // 光源の輝度
//                const Color emission = bodies[_hit.idx].getEmission();
//                //out_color += hitBody.getKd().cwiseProduct(emission) / kd;   // 重点的サンプリング済(ニュートン法)
//            }
//            else{
//                //out_color += hitBody.getKd().cwiseProduct(trace(_ray, _hit)) / kd;
//            }
        }
        else if(r < kd + ks) {
            Ray _ray; RayHit _hit;

            /// Marschner
            //marschnerSampleHair(ray, hit.point, hitBody.cylinder.axis, _ray);

            hitScene(_ray, _hit);

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

double Renderer::FresnelFunction(double cosTheta_i, double eta) const {
    cosTheta_i = std::clamp(cosTheta_i, -1.0, 1.0);
    // 髪の毛の内部から出射の場合
    if(cosTheta_i < 0){
        eta = 1 / eta;  // フレネルを式でnでまとめる
        cosTheta_i = -cosTheta_i;
    }

    double sinTheta_i = 1 - sqrt(cosTheta_i);
    double sinTheta_t = sinTheta_i / sqrt(eta);
    if(sinTheta_t >= 1){
        return 1.0;
    }
    double cosTheta_t = std::sqrt(1- sinTheta_t);

    double r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    double r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (sqrt(r_parl) + sqrt(r_perp)) / 2;
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
    Eigen::Vector3d wi = -in_Ray.dir;

    // w, v平面への投影ベクトルを計算する
    Eigen::Vector3d projected_on_wv = wi.dot(w) * w + wi.dot(v) * v;

    // 2.1. 出射角theta_oの計算（w, v平面への垂直成分を基に）
    //double theta_r = acos(neg_in_dir.dot(u)); // w軸に対する傾き
    double theta_r = acos(wi.dot(projected_on_wv));

    // 2.2. 出射角phi_oの計算（w, v平面上での投影方向）
    double phi_r = atan2(projected_on_wv.dot(v), projected_on_wv.dot(w)); // w, v平面での角度

    // 2.3. theta_iを-π/2 ~ +π/2の範囲に調整
    if (wi.dot(w) < 0) {
        // -in_Rayがw, v平面で-u方向にある場合、theta_iは負になるべき
        theta_r = -theta_r;
    }

    double sinTheta_r = sin(theta_r);
    double cosTheta_r = sqrt(1 - sqrt(sinTheta_r));

    Ray _ray; RayHit _hit;
//    hitScene(_ray, _hit);
//    double gamma_i =



    /// MとNによる出射角の決定
    /// ・・・
    /// 今は鏡面反射
    const double theta_i = - PI/2 - theta_r;
    const double phi_i = phi_r;

    /// theta, phiから出射ベクトルを計算
    const double _x = sin(theta_i) * sin(phi_i);    ///cos,sin入れ替えた
    const double _y = cos(theta_i);
    const double _z = sin(theta_i) * cos(phi_i);

    /// ローカルベクトルからグローバルのベクトルに変換
    out_Ray.dir = _x * w + _y * u + _z * v;

    out_Ray.org = incidentPoint;
}

void Renderer::marschnerSampleHair(const Ray &in_Ray, const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &axis, Ray &out_Ray, bool recordAngles, RaySampleData* rec) {
    /// axis(u)の方向をy軸とした正規直交基底を作る (w, u, v)
    Eigen::Vector3d w, v;
    Eigen::Vector3d u = axis;
    computeLocalFrame(u, w, v);

    // 入射光ベクトルをローカル基底に変換（w, v平面への投影を計算する）
    Eigen::Vector3d wi = -in_Ray.dir;

    // w, v平面への投影ベクトルを計算する
    Eigen::Vector3d projected_on_wv = (wi.dot(w) * w + wi.dot(v) * v).normalized();

    // 2.1. 出射角theta_oの計算（w, v平面への垂直成分を基に）
    //double theta_r = acos(neg_in_dir.dot(u)); // w軸に対する傾き
    double theta_i = std::acos(wi.dot(projected_on_wv));

    // 2.2. 出射角phi_oの計算（w, v平面上での投影方向）
    double phi_i = std::atan2(projected_on_wv.dot(v), projected_on_wv.dot(w)); // w, v平面での角度

    // 2.3. theta_iを-π/2 ~ +π/2の範囲に調整
    if (wi.dot(w) < 0) {
        // -in_Rayがw, v平面で-u方向にある場合、theta_iは負になるべき
        theta_i = -theta_i;
    }

    double sinTheta_i = sin(theta_i);
    double cosTheta_i = sqrt(1 - sqrt(sinTheta_i));

    Ray _ray; RayHit _hit;
//    hitScene(_ray, _hit);
//    double gamma_i =



    /// MとNによる出射角の決定
    /// ・・・
    /// 今は鏡面反射
    const double theta_o = - PI/2 - theta_i;
    const double phi_o = phi_i;

    /// theta, phiから出射ベクトルを計算
    const double _x = sin(theta_o) * sin(phi_o);    ///cos,sin入れ替えた
    const double _y = cos(theta_o);
    const double _z = sin(theta_o) * cos(phi_o);

    /// ローカルベクトルからグローバルのベクトルに変換
    out_Ray.dir = _x * w + _y * u + _z * v;

    out_Ray.org = incidentPoint;
}


void Renderer::diffuseSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray) const {
    /// normalの方向をy軸とした正規直交基底を作る (u, normal, v)
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    int count = 0;  // 収束しなかった回数

    try {
        const double theta = newton_method();
        const double phi = 2 * PI * rand();
        //const double theta = acos(1 - 2 * rand());  // sinのみのに沿わせる

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
    //double x = y * PI;

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
    ++exceptionCount; // カウントを増やす
    throw std::runtime_error("Newton method did not converge within the given max_iter.");
}

void Renderer::printExceptionCount() const {
    std::cout << "Number of Newton method convergence failures: " << exceptionCount << std::endl;
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

void Renderer::diffuseSample(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, bool recordAngles, RaySampleData* rec) const {
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

    if (recordAngles && rec) {
        rec->outgoingTheta = theta;
        rec->outgoingPhi   = phi;
    }
}

void Renderer::diffuseSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, bool recordAngles, RaySampleData* rec) const {
    /// normalの方向をy軸とした正規直交基底を作る (u, normal, v)
    Eigen::Vector3d u, v;
    computeLocalFrame(normal, u, v);

    int count = 0;  // 収束しなかった回数

    try {
        double theta = newton_method();
        // ↑背面へのケースがあったため修正
        if (theta > M_PI/2)                      // 背面→前面へ折り返し
            theta = M_PI - theta;
        const double phi = 2 * PI * rand();

        /// theta, phiから出射ベクトルを計算
        const double _x = sin(theta) * cos(phi);
        const double _y = cos(theta);
        const double _z = sin(theta) * sin(phi);

        /// ローカルベクトルからグローバルのベクトルに変換
        out_Ray.dir = _x * u + _y * normal + _z * v;

        out_Ray.org = incidentPoint;

        if (recordAngles && rec) {
            rec->outgoingTheta = theta;
            rec->outgoingPhi   = phi;
        }

    } catch (const std::exception &e) {
        std::cerr << "エラー: " << e.what() << std::endl;
    }
}

void Renderer::specularSampleHair(const Eigen::Vector3d &incidentPoint, const Eigen::Vector3d &normal, Ray &out_Ray, const double & n, bool recordAngles, RaySampleData* rec) const {
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

    /// theta, phiから出射ベクトルを計算
    const double _x = sin(theta) * cos(phi);
    const double _y = cos(theta);
    const double _z = sin(theta) * sin(phi);

    /// ローカルベクトルからグローバルのベクトルに変換
    out_Ray.dir = _x * u + _y * normal + _z * v;
    out_Ray.org = incidentPoint;

    if (recordAngles && rec) {
        rec->outgoingTheta = theta;
        rec->outgoingPhi   = phi;
    }
}

// w = 法線 n
void Renderer::worldToLocalAngles(const Eigen::Vector3d& n, const Eigen::Vector3d& dir, double& theta, double& phi) const
{
    Eigen::Vector3d u, v;          // u,v,n で正規直交基底
    computeLocalFrame(n, u, v);    // 既存関数

    double cosT = std::clamp(dir.dot(n), -1.0, 1.0);
    theta = std::acos(cosT);

    Eigen::Vector3d proj = (dir - cosT * n);
    if (proj.norm() < 1e-8) {              // 極 (proj ≈0) は φ=0 扱い
        phi = 0.0;
    } else {
        proj.normalize();
        phi = std::atan2(proj.dot(v), proj.dot(u));
    }
}

