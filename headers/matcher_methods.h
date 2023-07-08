#ifndef MATCHER_METHODS_H
#define MATCHER_METHODS_H

#include <opencv2/core.hpp>
#include "utils.h"

void bruteForceHammingSorted(cv::Mat img1, cv::Mat img2, Result res);
int bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res);
void bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res, cv::Mat &final);

void checkType(cv::Mat &img1, cv::Mat &img2, Result &res);
void computeMinMaxCoordinates(cv::Mat &final, std::vector<cv::DMatch> &goodMatches, Result &res);

#endif // MATCHER_METHODS_H