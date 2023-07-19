#ifndef UTILS_H
#define UTILS_H

/*
Written by @nicolacalzone and @rickyvendra
*/

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "ImageProcessor.h"

enum FoodType
{
    Meat,
    Beans
};

enum SharpnessType
{
    LAPLACIAN,
    HIGHPASS
};

struct FoodData
{
    cv::Mat src;
    cv::Rect box;
    std::vector<std::string> labels;
    std::vector<int> ids;
    cv::Mat segmentArea;
};

struct FoodDataContainer
{
    cv::Mat src;
    cv::Mat segmentArea;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<std::string>> labels;
    std::vector<std::vector<int>> ids;
    std::vector<int> areas;
};

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
    double dist = 0;
    double matches = 0;
    FoodDataContainer originalBB;
    FoodDataContainer leftoverBB;
};

struct Result
{
    std::vector<cv::KeyPoint> kp1, kp2; // keypoints
    cv::Mat descriptor1, descriptor2;   // descriptors
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

struct Area
{
    int area;
    std::vector<int> ids;
    cv::Mat segmentedMask;
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
cv::Mat convertBGRtoCIELAB(const cv::Mat &bgrImage);
std::vector<cv::Mat> convertBGRtoCIELAB(const std::vector<cv::Mat> &bgrImages);
void removeDish(cv::Mat &shifted);
double computeArea(cv::Rect box);
double computeCircleArea(double radius);
void computeSegmentArea(SegmentAreas &sa);
void acceptCircles(cv::Mat &in, cv::Mat &mask, cv::Mat &temp,
                   cv::Vec3i &c, cv::Point &center, int radius,
                   std::vector<cv::Vec3f> &accepted_circles,
                   std::vector<int> &dishesMatches, std::vector<cv::Mat> &dishes);
bool areSameImage(const cv::Mat &image1, const cv::Mat &image2);

#endif // UTILS_H