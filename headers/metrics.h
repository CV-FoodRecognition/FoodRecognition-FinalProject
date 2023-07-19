#ifndef METRICS_H
#define METRICS_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include "utils.h"

float get_iou(const FoodData &ground_truth, const FoodData &pred);

float get_meaniou(std::vector<FoodData> &groundTruth, std::vector<FoodData> &predictions);

std::set<cv::Point> getUnionSegments(cv::Mat &S1, cv::Mat &S2);

std::set<cv::Point> getIntersectionSegments(cv::Mat &S1, cv::Mat &S2);

float calculatePixelRatio(int pixelsAfterimg, int pixelsBeforeimg);

#endif // METRICS_H