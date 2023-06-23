#ifndef SEGMENT_H
#define SEGMENT_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

void doHough(std::vector<cv::Mat> &dishes, cv::Mat &in, cv::Mat &in_gray);
void doMSER(std::vector<cv::Rect> &mser_bbox, cv::Mat shifted, Mat result);
void kmeansSegmentation(int k, cv::Mat &src);
cv::Mat meanShiftFunct(cv::Mat src);
void removeShadows(cv::Mat &src, cv::Mat &dst);
void removeBackground(cv::Mat &src);
cv::Scalar computeAvgColor(cv::Mat shifted, cv::Rect box);

#endif