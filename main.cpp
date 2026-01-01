#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap("videos/street.mp4");

    if (!cap.isOpened()) {
        std::cerr << "Error: cannot open video file\n";
        return -1;
    }

    cv::Mat frame;
    cap >> frame;

    if (frame.empty()) {
        std::cerr << "Error: first frame is empty\n";
        return -1;
    }

    cv::imwrite("first_frame.jpg", frame);

    cv::imshow("Image", frame);
    cv::waitKey(0);

    return 0;
}