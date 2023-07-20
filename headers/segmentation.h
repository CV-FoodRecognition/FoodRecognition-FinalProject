#ifndef SEGMENT_H
#define SEGMENT_H

/*
Written by @nicolacalzone and @rickyvendra
*/

#include "utils.h"
#include "compute_dish.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void removeShadows(cv::Mat &src, cv::Mat &dst);
void removeBackground(cv::Mat &src);
cv::Scalar computeAvgColorCIELAB(const cv::Mat &input);
int boundBreadLeftover(cv::Mat &input, std::vector<cv::Mat> &dishes,
                       cv::Mat &final, std::vector<FoodData> &boundingBoxes);
int enlargeBoxAndComputeArea(cv::Rect box, cv::Mat inHSV, cv::Mat t);

#endif