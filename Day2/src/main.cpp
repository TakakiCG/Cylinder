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
            //Body(Cylinder(1.0, 10, Eigen::Vector3d(5, 0, 30), - Eigen::Vector3d::UnitX()), Material(Color(0.2, 0.1, 0.05), 0.2, 0.0, 0.7, 150.0)),

            //Body(Cylinder(1.0, 20.0, Eigen::Vector3d(0, -10, 30), Eigen::Vector3d::UnitY()), Material(Color(1.0, 1.0, 1.0), 0.4, 0.0, 0.5, 150.0)),
            //Body(Sphere(1.0, , Eigen::Vector3d(0, -14.5, 0), Eigen::Vector3d::UnitY()), Material(codeToColor("#864A2B"), 0.8, 0.0, 0.2)),
    };

    const std::vector<Body> lights {
            //Body(Sphere(5, Eigen::Vector3d(0, 34.8, 10)), Material(codeToColor("#e597b2"), 1.0, 30))
//            Body(Sphere(5, Eigen::Vector3d(0, 5, 40)), Material(codeToColor("#e597b2"), 1.0, 50)),   // 手前
            //Body(Sphere(5, Eigen::Vector3d(0, 20, 25)), Material(codeToColor("#e597b2"), 1.0, 100)),    // 真上
            //Body(Sphere(2, Eigen::Vector3d(0, 5, 40)), Material(codeToColor("#e597b2"), 1.0, 500))

//            Body(Cylinder(1.0, 20, Eigen::Vector3d(-10, 15, 25), Eigen::Vector3d::UnitX()), Material(codeToColor("#e597b2"), 1.0, 100))

            Body(Cylinder(5.0, 30, Eigen::Vector3d(-15, 25, 30), Eigen::Vector3d::UnitX()), Material(codeToColor("#e597b2"), 1.0, 100)),    // 上#e597b2
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
    std::vector<Body> hairs = HairGenerator::generateHairs(5000, 5, 0.025, headCenter, headRadius);  // 0.05
    bodies.insert(bodies.end(), hairs.begin(), hairs.end());
    std::cout << "Hairs were generated." << std::endl;

//    std::vector<Body> hairs = HairGenerator::generateStraightHairsInLine(100, 0.05, Eigen::Vector3d(-5.0, 10.0, 25.0), 20.0);
//    bodies.insert(bodies.end(), hairs.begin(), hairs.end());

    const Eigen::Vector3d campos(0, 0, 80);
    const Eigen::Vector3d camdir = Eigen::Vector3d(0, 0, 0) - campos;

    //const Camera camera(campos, camdir, 540, 3.0 / 4.0, 60, 45);
    const Camera camera(campos, camdir, 540, 3.0 / 4.0, 25 , 45);    // vertical25

    /// 背景色はわかりやすく灰色
    const Renderer renderer(bodies, camera, Color(0.1, 0.1, 0.1));
    //const auto image = renderer.render().apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    //std::cout << "Rendered a Image1" << std::endl;

    const unsigned int samples = 100;
//    const unsigned int areas_n_samples = 10;
//    const unsigned int _samples = samples / areas_n_samples;
    //const auto image2 = renderer._directIlluminationRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();
//    const auto image2 = renderer._directIlluminationRender_anti_areas(samples, areas_n_samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    const auto image3 = renderer.passTracingRender(samples).apply_reinhard_extended_tone_mapping().apply_gamma_correction();

    std::cout << "Rendered a Image3" << std::endl;

//    image.save("sample_image_cylinder.png");
    //image2.save("rayTracing_sample_5000_KK_10000_r0.025_diff_newton.png");
    //image2.save("Phong_specular_ray_test_nonNorm.png");
    //image3.save("passTracing_sample_5000_KK_10000_kd02_ks07_n150.png");
    //image3.save("passTracing_test_ks05_150_newton.png");
    //image2.save("rayTracing_test_straight.png");
    //image3.save("visualize.png");
    image3.save("test.png");

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

cv::Mat createAcademicColorBar(int height,
                               int barWidth,
                               int textMargin,
                               double minValue = 0.0,
                               double maxValue = 255.0)
{
    // 返す画像は height x (barWidth + textMargin) のサイズ
    // カラー3チャネル (BGR) で真っ黒に初期化
    cv::Mat colorBar(height, barWidth + textMargin, CV_8UC3, cv::Scalar(0, 0, 0));

    // ---------------------------
    // 1. 左側に配置するグレースケールバーを作成
    // ---------------------------
    // グレースケール1チャネル (barWidth 分の横幅)
    cv::Mat grayBar(height, barWidth, CV_8UC1);

    // 下が0 (最小値) で上が255 (最大値) になるように
    for (int i = 0; i < height; ++i) {
        int value = static_cast<int>(
            ((height - 1 - i) / static_cast<double>(height - 1)) * 255.0
        );
        grayBar.row(i).setTo(value);
    }

    // Jetカラーマップを適用 (他のカラーマップでもOK)
    cv::Mat colorMapBar;
    cv::applyColorMap(grayBar, colorMapBar, cv::COLORMAP_JET);

    // colorBar の左側 (0,0) ～ (barWidth-1, height-1) にコピー
    colorMapBar.copyTo(colorBar(cv::Rect(0, 0, barWidth, height)));

    // ---------------------------
    // 2. 右側のテキスト・目盛り部分を白で塗りつぶす
    // ---------------------------
    cv::rectangle(colorBar,
                  cv::Point(barWidth, 0),               // テキスト領域の左上
                  cv::Point(barWidth + textMargin, height - 1), // テキスト領域の右下
                  cv::Scalar(255, 255, 255),           // 白
                  cv::FILLED);

    // ---------------------------
    // 3. メモリ (0, 中間値, 255) とラベル (テキスト) を描画
    // ---------------------------
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.45;
    int thickness = 1;

    // メモリ位置と値
    std::vector<std::pair<int, int>> ticks = {
        {height - 1, static_cast<int>(minValue)},       // 最下部 (0%)
        {height / 2, static_cast<int>((minValue + maxValue) / 2)}, // 中間値 (50%)
        {0, static_cast<int>(maxValue)}                // 最上部 (100%)
    };

    for (const auto &tick : ticks) {
        int y = tick.first;   // y座標
        int value = tick.second; // メモリ値

        // ラベル (整数値)
        std::ostringstream labelStream;
        labelStream << value;
        std::string label = labelStream.str();

        // テキストサイズを取得
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, fontFace, fontScale, thickness, &baseline);

        // テキストを描画する座標
        cv::Point textOrg;
        if (y == 0) { // 最上部（最大値）の場合、テキストを少し下げる
            textOrg = cv::Point(barWidth + 5, y + textSize.height);
        } else if (y == height - 1) { // 最下部（最小値）の場合、テキストを少し上げる
            textOrg = cv::Point(barWidth + 5, y - textSize.height / 2);
        } else { // 中間値はそのまま中央揃え
            textOrg = cv::Point(barWidth + 5, y + textSize.height / 2 - 2);
        }

        // 目盛りの線 (バー末端付近に短い横線)
        cv::line(colorBar,
                 cv::Point(barWidth - 5, y),       // 少し左にオフセット
                 cv::Point(barWidth,     y),       // バー右端
                 cv::Scalar(0, 0, 0),             // 黒い目盛り線
                 1);

        // テキストを描画
        cv::putText(colorBar,
                    label,
                    textOrg,
                    fontFace,
                    fontScale,
                    cv::Scalar(0, 0, 0),  // 黒い文字
                    thickness);
    }

    return colorBar;
}

//int main()
//{
//    // 1. 入力画像を読み込む (グレースケール)
//    cv::Mat inputImage = cv::imread("passTracing_sample_5000_KK_10000_kd02_ks07_n150.png", cv::IMREAD_GRAYSCALE);
//    if (inputImage.empty()) {
//        std::cerr << "Error: Could not open the image file!" << std::endl;
//        return -1;
//    }
//
//    // 2. 入力画像の最小値・最大値を取得
//    double minVal, maxVal;
//    cv::minMaxLoc(inputImage, &minVal, &maxVal);
//
//    // 3. 画像を 0～255 に正規化 (可視化用)
//    cv::Mat normalizedImage;
//    cv::normalize(inputImage, normalizedImage, 0, 255, cv::NORM_MINMAX, CV_8UC1);
//
//    // 4. ヒートマップを作成
//    cv::Mat heatmap;
//    cv::applyColorMap(normalizedImage, heatmap, cv::COLORMAP_JET);
//
//    // 5. カラーバーを作成 (範囲は0～255)
//    int barHeight = heatmap.rows;
//    int barWidth = 40;    // 実際のカラーグラデーション部分の幅
//    int textMargin = 70;  // 目盛り用マージン (テキスト描画領域)
//
//    cv::Mat colorBar = createAcademicColorBar(barHeight,
//                                              barWidth,
//                                              textMargin,
//                                              0,     // カラーバーの最小値
//                                              255);  // カラーバーの最大値
//
//    // 6. ヒートマップとカラーバーを横に結合
//    cv::Mat combined;
//    cv::hconcat(heatmap, colorBar, combined);
//
//    // 7. 結果を保存
//    cv::imwrite("heatmap_pass_kd02_ks07_n150.png", combined);
//
//    // 8. 結果を表示
//    cv::imshow("Heatmap with Academic Colorbar", combined);
//    cv::waitKey(0);
//
//    return 0;
//}

