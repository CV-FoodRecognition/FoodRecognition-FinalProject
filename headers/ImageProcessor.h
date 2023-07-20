#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

/*
Class written by @nicolacalzone and @rickyvendra
*/

#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"

class ImageProcessor
{
private:
    std::vector<cv::Mat> dishes;
    std::vector<int> dishesMatches;
    std::vector<cv::Rect> mser_bbox;
    std::vector<std::vector<cv::Point>> regions;
    std::vector<int> radia;
    std::vector<cv::Vec3f> acceptedCircles;

public:
    ImageProcessor() {}

    void doHough(cv::Mat &in);
    void doMSER(cv::Mat &shifted, cv::Mat &result);
    cv::Mat kmeansSegmentation(int k, cv::Mat &src);
    std::vector<cv::Mat> removeDish(const std::vector<cv::Mat> &src);
    std::vector<cv::Mat> &getDishes() { return dishes; }
    std::vector<int> &getDishesMatches() { return dishesMatches; }
    std::vector<cv::Rect> &getMserBbox() { return mser_bbox; }
    std::vector<int> &getRadius() { return radia; }
    std::vector<cv::Vec3f> &getAcceptedCircles() { return acceptedCircles; }
};

// Helper Function for Hough
bool isInside(std::vector<cv::Vec3f> circles, cv::Point center);

#endif
