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

const float IOUthresh = 0.5;

enum SharpnessType
{
    LAPLACIAN,
    HIGHPASS
};

struct FoodData
{
    cv::Rect box;
    std::string label;
    int id;
    cv::Mat segmentArea;
    int area;
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
    double dist = -1;
    double matches = -1;
    bool empty = false;
};

struct SegmentCouple
{
    int id = -1;
    cv::Mat segmentOriginal;
    cv::Mat segmentLeftover;
};

// Used as Result from SIFT and SURF
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

cv::Mat getYellowArea(cv::Mat &segmented);
cv::Mat getBlueArea(cv::Mat &segmented);
void addFood(int size, std::string fileName, std::string label, int id,
             std::string path, std::vector<foodTemplate> &templates);
bool isInsideCircle(cv::Vec3i c, int x, int y);
void showImg(std::string title, cv::Mat image);
void concatShowImg(std::string title, cv::Mat image1, cv::Mat image2);
void sharpenImg(cv::Mat &src, SharpnessType t);
cv::Mat convertGray(cv::Mat &src);
cv::Mat convertBGRtoCIELAB(const cv::Mat &bgrImage);
std::vector<cv::Mat> convertBGRtoCIELAB(const std::vector<cv::Mat> &bgrImages);
void removeDish(cv::Mat &src);
double computeArea(cv::Rect box);
double computeCircleArea(double radius);
void computeSegmentArea(SegmentAreas &sa);
void acceptCircles(cv::Mat &in, cv::Mat &mask, cv::Mat &temp,
                   cv::Vec3i &c, cv::Point &center, int radius,
                   std::vector<cv::Vec3f> &accepted_circles,
                   std::vector<int> &dishesMatches, std::vector<cv::Mat> &dishes);
bool areSameImage(const cv::Mat &image1, const cv::Mat &image2);

#endif // UTILS_H