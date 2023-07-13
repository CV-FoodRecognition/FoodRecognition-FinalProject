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
cv::Scalar computeAvgColorHSV(cv::Mat shifted);
int computeBestDish(foodTemplate food, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches);
void boundPasta(cv::Mat &dish, cv::Mat &final, std::string label,
                std::vector<int> forbidden, int max_key, std::vector<BoundingBox> &boundingBoxes);
void boundBread(cv::Mat &input, std::vector<cv::Mat> &dishes,
                cv::Mat &final, std::vector<BoundingBox> &boundingBoxes);
void drawBoundingBoxes(cv::Mat &final, std::vector<BoundingBox> &boundingBoxes);

#endif