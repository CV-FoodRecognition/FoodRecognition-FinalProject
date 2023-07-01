#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct Result
{
    std::vector<cv::KeyPoint> kp1, kp2; // keypoints
    cv::Mat descriptor1, descriptor2;   // descriptors
};

enum SharpnessType
{
    LAPLACIAN,
    HIGHPASS
};

enum FoodType
{
    Meat,
    Beans
};

struct BoxLabel
{
    cv::Rect mser_box;
    FoodType label;
};

std::string enumToString(FoodType label);
bool isInsideCircle(cv::Vec3i c, int x, int y);
void showImg(std::string title, cv::Mat image);
void sharpenImg(cv::Mat &src, SharpnessType t);
cv::Mat convertGray(cv::Mat &src);
void removeDish(cv::Mat &shifted);
double computeArea(cv::Rect box);

#endif // UTILS_H