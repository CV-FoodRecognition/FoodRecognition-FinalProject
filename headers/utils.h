#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "ImageProcessor.h"

struct foodTemplate
{
    std::vector<cv::Mat> foodTemplates;
    std::string label;
    int id;
};

struct Couple
{
    cv::Mat original;
    cv::Mat leftover;
};

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

struct PassedStruct
{
    cv::Mat p1;
    std::string p2;
};

struct SegmentAreas
{
    cv::Mat p1;
    int areaYellow;
    int areaBlue;
    int areaRed;
    int areaGreen;
    int areaBlack;
    cv::Point topLeft;
    cv::Point bottomRight;
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
    cv::Scalar averageBoxColor;
    double areaBox;
};

void addFood(int size, std::string fileName, std::string label, int id,
             std::string path, std::vector<foodTemplate> &templates);
std::string enumToString(FoodType label);
bool isInsideCircle(cv::Vec3i c, int x, int y);
void showImg(std::string title, cv::Mat image);
void concatShowImg(std::string title, cv::Mat image1, cv::Mat image2);
void sharpenImg(cv::Mat &src, SharpnessType t);
cv::Mat convertGray(cv::Mat &src);
void removeDish(cv::Mat &shifted);
double computeArea(cv::Rect box);
double computeCircleArea(double radius);
void computeSegmentArea(SegmentAreas &sa);
void acceptCircles(cv::Mat &in, cv::Mat &mask, cv::Mat &temp,
                   cv::Vec3i &c, cv::Point &center, int radius,
                   std::vector<cv::Vec3f> &accepted_circles,
                   std::vector<int> &dishesMatches, std::vector<cv::Mat> &dishes);

#endif // UTILS_H