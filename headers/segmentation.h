#ifndef SEGMENT_H
#define SEGMENT_H

#include "../headers/utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void doHough(std::vector<cv::Mat> &dishes, std::vector<int> &dishesMatches, Mat &in);
void doMSER(std::vector<cv::Rect> &mser_bbox, cv::Mat shifted, Mat result);
cv::Mat kmeansSegmentation(int k, cv::Mat &src);
cv::Mat meanShiftFunct(cv::Mat src);
void removeShadows(cv::Mat &src, cv::Mat &dst);
void removeBackground(cv::Mat &src);
cv::Scalar computeAvgColor(cv::Mat shifted, cv::Rect box);
// bool isInside(std::vector<cv::Vec3f> circles, cv::Point center);
#endif