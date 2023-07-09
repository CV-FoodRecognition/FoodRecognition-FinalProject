#ifndef SEGMENT_H
#define SEGMENT_H

#include "../headers/utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void removeShadows(cv::Mat &src, cv::Mat &dst);
void removeBackground(cv::Mat &src);
cv::Scalar computeAvgColor(cv::Mat shifted, cv::Rect box);
cv::Scalar computeAvgColor(cv::Mat shifted);

#endif