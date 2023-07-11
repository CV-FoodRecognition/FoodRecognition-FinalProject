#ifndef MATCHER_METHODS_H
#define MATCHER_METHODS_H

#include <opencv2/core.hpp>
#include <array>
#include <iostream>
#include "utils.h"
#include "descriptor_methods.h"

void bruteForceHammingSorted(cv::Mat img1, cv::Mat img2, Result res);
int bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res);
void bruteForceKNN(cv::Mat img1, foodTemplate food, cv::Mat dish, cv::Mat &final);
void checkType(cv::Mat &img1, cv::Mat &img2, Result &res);
void computeMinMaxCoordinates(cv::Mat &final, std::vector<cv::DMatch> &goodMatches, Result &res);

#endif // MATCHER_METHODS_H