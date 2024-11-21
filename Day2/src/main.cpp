#include <iostream>
#include <chrono>
#include "Body.h"
#include "Camera.h"
#include "Renderer.h"
#include "HairGenerator.h"

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

void sample() {
    /// bodiesに光源を追加
    const std::vector<Body> bodies =  {
            Body(Sphere(1.0, Eigen::Vector3d::Zero()), Material(Color(1, 0.1, 0.1), 0.8)),
            Body(Sphere(1.0, Eigen::Vector3d(0, 3, 0)), Material(Color(0.1, 1, 0.1), 0.8)),
            Body(Sphere(1.0, Eigen::Vector3d(0, -3, 0)), Material(Color(0.1, 0.1, 1), 0.8)),
            Body(Sphere(2.0, Eigen::Vector3d(0, 10, 10)), Material(Color(1, 1, 1), 0.8, 10)),
            Body(Cylinder(1.0, 2.0, Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY()), Material(Color(1, 0.1, 0.1), 0.8)),
            Body(Cylinder(1.0, 2.0, Eigen::Vector3d(0, 3, 0), Eigen::Vector3d::UnitY()), Material(Color(0.1, 1, 0.1), 0.8)),
            Body(Cylinder(1.0, 2.0, Eigen::Vector3d(0, -3, 0), Eigen::Vector3d::UnitY()), Material(Color(0.1, 0.1, 1), 0.8)),
            Body(Cylinder(2.0, 4.0, Eigen::Vector3d(0, 10, 10), Eigen::Vector3d::UnitY()), Material(Color(1, 1, 1), 0.8, 10)),
    };

    const Eigen::Vector3d campos(0, 10, 100);
    const Eigen::Vector3d camdir = Eigen::Vector3d(0, 0, 0) - campos;

    const Camera camera(campos, camdir, 320, 9.0 / 16.0, 5);

    /// 背景色はわかりやすく灰色
    const Renderer renderer(bodies, camera, Color(0.1, 0.1, 0.1));
    const unsigned int samples = 10000;
    const auto image1 = renderer.render();
    const auto image2 = renderer.directIlluminationRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    image1.save("sample_image_cylinder.png");
    image2.apply_reinhard_extended_tone_mapping().save("sample_cylinder.png");
}

void roomRenderingSample() {
    const auto room_r = 1e5;
    const auto floor_color = codeToColor("#f9c89b");
    const std::vector<Body> room_walls {
            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitX()), Material(Color(0.05, 0.05, 0.05), 0.0, 0.0)),
            Body(Sphere(room_r, -(room_r - 30) * Eigen::Vector3d::UnitX()), Material(Color(0.05, 0.05, 0.05), 0.0, 0.0)),
            Body(Sphere(room_r, (room_r - 30) * Eigen::Vector3d::UnitY()), Material(Color(0.05, 0.05, 0.05), 0.0, 0.0)),
            Body(Sphere(room_r, -(room_r - 40) * Eigen::Vector3d::UnitY()), Material(Color(0.05, 0.05, 0.05), 0.0, 0.0)),
            Body(Sphere(room_r, (room_r - 5) * Eigen::Vector3d::UnitZ()), Material(Color(0.05, 0.05, 0.05), 0.05, 0.0)),
    };

    std::vector<Body> bodies {
            //Body(Cylinder(1.0, 40.0, Eigen::Vector3d(0, -14.5, 0), Eigen::Vector3d::UnitY()), Material(codeToColor("#864A2B"), 0.8, 0.0, 0.2)),
    };

    const std::vector<Body> lights {
            //Body(Sphere(5, Eigen::Vector3d(0, 34.8, 10)), Material(codeToColor("#e597b2"), 1.0, 30))
            Body(Sphere(5, Eigen::Vector3d(0, 20, 40)), Material(codeToColor("#e597b2"), 1.0, 30)),
            //Body(Sphere(2, Eigen::Vector3d(0, 5, 40)), Material(codeToColor("#e597b2"), 1.0, 500))
    };

    for(const auto & room_wall : room_walls) {
        bodies.push_back(room_wall);
    }

    for(const auto & light : lights) {
        bodies.push_back(light);
    }

    Eigen::Vector3d headCenter(0.0, 10.0, 30.0); // 頭の中心位置
    double headRadius = 5.0; // 頭の半径
    Material headMaterial(codeToColor("#717375"), 0.05);
    Body head(Sphere(headRadius, headCenter), headMaterial);
    bodies.push_back(head);

    /// 髪の毛を生成して bodies に追加
    std::vector<Body> hairs = HairGenerator::generateHairs(10, 5, 0.05, headCenter, headRadius);
    bodies.insert(bodies.end(), hairs.begin(), hairs.end());


    const Eigen::Vector3d campos(0, 0, 80);
    const Eigen::Vector3d camdir = Eigen::Vector3d(0, 0, 0) - campos;

    const Camera camera(campos, camdir, 540, 4.0 / 3.0, 60, 45);

    /// 背景色はわかりやすく灰色
    const Renderer renderer(bodies, camera, Color(0.1, 0.1, 0.1));
    const auto image = renderer.render().apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    const unsigned int samples = 1e3;
    const auto image2 = renderer._directIlluminationRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    image.save("sample_image_cylinder.png");
    image2.save("sample_1000_cylinder_MAR10.png");
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
