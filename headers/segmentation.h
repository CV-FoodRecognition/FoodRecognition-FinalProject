#ifndef SEGMENT_H
#define SEGMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void kmeansSegmentation(int k, cv::Mat &src);
cv::Mat meanShiftFunct(cv::Mat src);

#endif