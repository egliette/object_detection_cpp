#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <iostream>

int main() {
    cv::VideoCapture cap("videos/street.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video file\n";
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "FPS: " << fps << " ms\n";

    cv::Mat frame;
    int frameCount = 0;

    cv::namedWindow("Video", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video", 1280, 720);
    
    auto lastTime = std::chrono::steady_clock::now();
    int targetFrameMs = static_cast<int>(1000.0  / fps);
    auto lastReportTime = std::chrono::steady_clock::now();
    auto lastFrameTime = lastReportTime;

    while (true) {
        cap >> frame;
        cv::Point topLeft(1319, 503);
        cv::Point bottomRight(1392, 633);
        cv::rectangle(frame, topLeft, bottomRight, cv::Scalar(0, 255, 0), 2);

        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        
        frameCount++;
        cv::imshow("Video", frame);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            now - lastTime
        ).count();

        if (elapsed >= 5) {
            std::cout << frameCount << std::endl;
            lastTime = now;
        }

        cv::waitKey(1);
        
        auto frameEnd = std::chrono::steady_clock::now();
        auto frameElapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            frameEnd - lastFrameTime
        ).count();

        int sleepMs = targetFrameMs - static_cast<int>(frameElapsedMs);
        if (sleepMs > 0) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(sleepMs)
            );
        }

        lastFrameTime = std::chrono::steady_clock::now();
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}