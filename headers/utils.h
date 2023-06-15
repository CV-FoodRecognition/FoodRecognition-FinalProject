#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

std::vector<cv::Mat> analyzeImages(const std::string &filepathJPG0, const std::string &filepathJPG1);
bool isInsideCircle(cv::Vec3i c, int x, int y);
void showImg(std::string title, cv::Mat image);

struct Result
{
    std::vector<cv::KeyPoint> kp1, kp2; // keypoints
    cv::Mat descriptor1, descriptor2;   // descriptors
};

#endif // UTILS_H