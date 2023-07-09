#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <vector>
#include <opencv2/opencv.hpp>

class ImageProcessor
{
private:
    std::vector<cv::Mat> dishes;
    std::vector<int> dishesMatches;
    std::vector<cv::Rect> mser_bbox;
    std::vector<std::vector<cv::Point>> regions;
    std::vector<int> radia;

public:
    ImageProcessor() {}

    void doHough(cv::Mat &in);
    void doMSER(cv::Mat &shifted, cv::Mat &result);
    cv::Mat kmeansSegmentation(int k, cv::Mat &src);

    const std::vector<cv::Mat> &getDishes() const { return dishes; }
    const std::vector<int> &getDishesMatches() const { return dishesMatches; }
    const std::vector<cv::Rect> &getMserBbox() const { return mser_bbox; }
    const std::vector<int> &getRadius() const { return radia; }
};

// Helper Function for Hough
bool isInside(std::vector<cv::Vec3f> circles, cv::Point center);

#endif
