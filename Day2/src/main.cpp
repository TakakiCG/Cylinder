#include <iostream>
#include <chrono>
#include "Body.h"
#include "Camera.h"
#include "Renderer.h"
#include "HairGenerator.h"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void intersectTest() {
    const Sphere sphere(1, Eigen::Vector3d::Zero());
    const Cylinder cylinder(1.0, 2.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY());
    Ray ray(Eigen::Vector3d(0, 0, 10), Eigen::Vector3d(0, 0, -1));
    RayHit hit;
    sphere.hit(ray, hit);
    cylinder.hit(ray, hit);

    std::cout << "t:\t" << hit.t << std::endl;
    std::cout << "normal:\t(" << hit.normal.transpose() << ")" << std::endl;
}

//void sample() {
//    /// bodiesに光源を追加
//    const std::vector<Body> bodies =  {
//            Body(Sphere(1.0, Eigen::Vector3d::Zero()), Material(Color(1, 0.1, 0.1), 0.8)),
//            Body(Sphere(1.0, Eigen::Vector3d(0, 3, 0)), Material(Color(0.1, 1, 0.1), 0.8)),
//            Body(Sphere(1.0, Eigen::Vector3d(0, -3, 0)), Material(Color(0.1, 0.1, 1), 0.8)),
//            Body(Sphere(2.0, Eigen::Vector3d(0, 10, 10)), Material(Color(1, 1, 1), 0.8, 10)),
//            Body(Cylinder(1.0, 2.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY()), Material(Color(1, 0.1, 0.1), 0.8)),
//            Body(Cylinder(1.0, 2.0, Eigen::Vector3d(0, 3, 0), Eigen::Vector3d::UnitY()), Material(Color(0.1, 1, 0.1), 0.8)),
//            Body(Cylinder(1.0, 2.0, Eigen::Vector3d(0, -3, 0), Eigen::Vector3d::UnitY()), Material(Color(0.1, 0.1, 1), 0.8)),
//            Body(Cylinder(2.0, 4.0, Eigen::Vector3d(0, 10, 10), Eigen::Vector3d::UnitY()), Material(Color(1, 1, 1), 0.8, 10)),
//    };
//
//    const Eigen::Vector3d campos(0, 10, 100);
//    const Eigen::Vector3d camdir = Eigen::Vector3d(0, 0, 0) - campos;
//
//    const Camera camera(campos, camdir, 320, 9.0 / 16.0, 5);
//
//    /// 背景色はわかりやすく灰色
//    const Renderer renderer(bodies, camera, Color(0.1, 0.1, 0.1));
//    const unsigned int samples = 10000;
//    const auto image1 = renderer.render();
//    const auto image2 = renderer.directIlluminationRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();
//
//    image1.save("sample_image_cylinder.png");
//    image2.apply_reinhard_extended_tone_mapping().save("sample_cylinder.png");
//}

void roomRenderingSample() {
    const auto room_r = 1e5;
    const auto floor_color = codeToColor("#f9c89b");
    const std::vector<Body> room_walls {
            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitX()), Material(codeToColor("#f9c89b"), 0.4, 0.0)),
            Body(Sphere(room_r, -(room_r - 30) * Eigen::Vector3d::UnitX()), Material(codeToColor("#f9c89b"), 0.4, 0.0)),
            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitY()), Material(codeToColor("#f9c89b"), 0.4, 0.0)),
            Body(Sphere(room_r, -(room_r - 40) * Eigen::Vector3d::UnitY()), Material(codeToColor("#f9c89b"), 0.4, 0.0)),
            Body(Sphere(room_r, (room_r - 5) * Eigen::Vector3d::UnitZ()), Material(codeToColor("#f9c89b"), 0.4, 0.0)),

// Phongエネルギー保存確認
//            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitX()), Material(Color(1.0, 1.0, 1.0), 1.0, 1.0)),   // 右
//            Body(Sphere(room_r, -(room_r - 30) * Eigen::Vector3d::UnitX()), Material(Color(1.0, 1.0, 1.0), 1.0, 1.0)),  // 左
//            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitY()), Material(Color(1.0, 1.0, 1.0), 1.0, 1.0)),   // 上
//            Body(Sphere(room_r, -(room_r - 40) * Eigen::Vector3d::UnitY()), Material(Color(1.0, 1.0, 1.0), 1.0, 1.0)),  // 下
//            Body(Sphere(room_r, (room_r - 5) * Eigen::Vector3d::UnitZ()), Material(Color(1.0, 1.0, 1.0), 1.0, 1.0)),    // 奥
//            Body(Sphere(room_r, (room_r + 35) * Eigen::Vector3d::UnitZ()), Material(Color(1.0, 1.0, 1.0), 1.0, 1.0)),   // 手前

//            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitX()), Material(Color(1, 1, 1), 0.0, 0.0)),
//            Body(Sphere(room_r, -(room_r - 30) * Eigen::Vector3d::UnitX()), Material(Color(1, 1, 1), 0.0, 0.0)),
//            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitY()), Material(Color(1, 1, 1), 0.0, 0.0)),
//            Body(Sphere(room_r, -(room_r - 40) * Eigen::Vector3d::UnitY()), Material(Color(1, 1, 1), 0.0, 0.0)),
//            Body(Sphere(room_r, (room_r - 5) * Eigen::Vector3d::UnitZ()), Material(Color(1, 1, 1), 0.0, 0.0)),
    };

    std::vector<Body> bodies {
            //Body(Cylinder(1.0, 20.0, Eigen::Vector3d(0, -10, 30), Eigen::Vector3d::UnitY()), Material(Color(0.2, 0.1, 0.05), 0.2, 0.0, 0.7, 150.0)),

            //Body(Cylinder(1.0, 20.0, Eigen::Vector3d(0, -10, 30), Eigen::Vector3d::UnitY()), Material(Color(1.0, 1.0, 1.0), 0.4, 0.0, 0.5, 150.0)),
            //Body(Sphere(1.0, , Eigen::Vector3d(0, -14.5, 0), Eigen::Vector3d::UnitY()), Material(codeToColor("#864A2B"), 0.8, 0.0, 0.2)),
    };

    const std::vector<Body> lights {
            //Body(Sphere(5, Eigen::Vector3d(0, 34.8, 10)), Material(codeToColor("#e597b2"), 1.0, 30))
//            Body(Sphere(5, Eigen::Vector3d(0, 5, 40)), Material(codeToColor("#e597b2"), 1.0, 50)),   // 手前
            //Body(Sphere(5, Eigen::Vector3d(0, 20, 25)), Material(codeToColor("#e597b2"), 1.0, 100)),    // 真上
            //Body(Sphere(2, Eigen::Vector3d(0, 5, 40)), Material(codeToColor("#e597b2"), 1.0, 500))

//            Body(Cylinder(1.0, 20, Eigen::Vector3d(-10, 15, 25), Eigen::Vector3d::UnitX()), Material(codeToColor("#e597b2"), 1.0, 100))

            Body(Cylinder(5.0, 30, Eigen::Vector3d(-15, 25, 30), Eigen::Vector3d::UnitX()), Material(codeToColor("#e597b2"), 1.0, 100)),    // 上
            //Body(Cylinder(1.0, 30, Eigen::Vector3d(-15, 3, 40), Eigen::Vector3d::UnitZ()), Material(codeToColor("#e597b2"), 1.0, 100)),     // 左
            //Body(Cylinder(1.0, 30, Eigen::Vector3d(15, 3, 40), Eigen::Vector3d::UnitZ()), Material(codeToColor("#e597b2"), 1.0, 100))       // 右
            Body(Cylinder(1.0, 30, Eigen::Vector3d(-15, 5.5, 40), Eigen::Vector3d::UnitX()), Material(codeToColor("#e597b2"), 1.0, 50)),    // 手前

            //Body(Cylinder(5.0, 30, Eigen::Vector3d(-15, 25, 30), Eigen::Vector3d::UnitX()), Material(codeToColor("#e597b2"), 1.0, 100)),    // 上・球
            //Body(Sphere(3, Eigen::Vector3d(0, 4, 50)), Material(codeToColor("#e597b2"), 1.0, 50))  // 手前・球


    };

    for(const auto & room_wall : room_walls) {
        bodies.push_back(room_wall);
    }

    for(const auto & light : lights) {
        bodies.push_back(light);
    }


    Eigen::Vector3d headCenter(0.0, 6.0, 25.0); // 頭の中心位置
    double headRadius = 5.0; // 頭の半径
    Material headMaterial(codeToColor("#E5C6AA"), 0.3);
    Body head(Sphere(headRadius, headCenter), headMaterial);
    bodies.push_back(head);

    /// 髪の毛を生成して bodies に追加
    std::cout << "Generating hairs ..." << std::endl;
    std::vector<Body> hairs = HairGenerator::generateHairs(10000, 5, 0.025, headCenter, headRadius);  // 0.05
    bodies.insert(bodies.end(), hairs.begin(), hairs.end());
    std::cout << "Hairs were generated." << std::endl;

//    std::vector<Body> hairs = HairGenerator::generateStraightHairsInLine(100, 0.05, Eigen::Vector3d(-5.0, 10.0, 25.0), 20.0);
//    bodies.insert(bodies.end(), hairs.begin(), hairs.end());

    const Eigen::Vector3d campos(0, 0, 80);
    const Eigen::Vector3d camdir = Eigen::Vector3d(0, 0, 0) - campos;

//    const Camera camera(campos, camdir, 540, 3.0 / 4.0, 25, 45);
    const Camera camera(campos, camdir, 540, 3.0 / 4.0, 25 , 45);    // vertical25

    /// 背景色はわかりやすく灰色
    const Renderer renderer(bodies, camera, Color(0.1, 0.1, 0.1));
    //const auto image = renderer.render().apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    //std::cout << "Rendered a Image1" << std::endl;

    const unsigned int samples = 5000;
//    const unsigned int areas_n_samples = 10;
//    const unsigned int _samples = samples / areas_n_samples;
    //const auto image2 = renderer._directIlluminationRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();
//    const auto image2 = renderer._directIlluminationRender_anti_areas(samples, areas_n_samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    const auto image3 = renderer.passTracingRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    std::cout << "Rendered a Image3" << std::endl;

//    image.save("sample_image_cylinder.png");
    //image2.save("rayTracing_sample_5000_KK_10000_r0.025_diff_newton.png");
    //image2.save("Phong_specular_ray_test_nonNorm.png");
    image3.save("passTracing_sample_5000_KK_10000_kd00_ks07_n150.png");
    //image3.save("passTracing_test_ks05_150_newton.png");
    //image2.save("rayTracing_test_straight.png");
    //image3.save("M_test.png");

}

int main() {
    auto start = std::chrono::steady_clock::now();
    std::cout << "Hello, World!" << std::endl;
    intersectTest();

    roomRenderingSample();

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    auto seconds = duration - minutes;
    std::cout << "経過時間: " << minutes.count() << " 分 " << seconds.count() << " 秒" << std::endl;
    return 0;
}

//int main(int argc, char** argv) {
//    // 画像を読み込む
//    Mat image1 = imread("passTracing_sample_1000_KK_5000_nowall.png", IMREAD_GRAYSCALE);
//    Mat image2 = imread("rayTracing_sample_1000_KK_5000.png", IMREAD_GRAYSCALE);
//
//    if (image1.empty() || image2.empty()) {
//        cout << "Could not open or find the images!" << endl;
//        return -1;
//    }
//
//    // 輝度差を計算する
//    Mat diff;
//    absdiff(image1, image2, diff);
//
//    // 結果を表示する
//    imshow("Image 1", image1);
//    imshow("Image 2", image2);
//    imshow("Difference", diff);
//
//    // 結果を保存する
//    imwrite("difference.jpg", diff);
//
//    waitKey(0);
//    return 0;
//}
