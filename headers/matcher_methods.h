#ifndef MATCHER_METHODS_H
#define MATCHER_METHODS_H

/*
Written by @nicolacalzone and @rickyvendra
*/

#include <opencv2/core.hpp>
#include <array>
#include <iostream>
#include "utils.h"
#include "Dish.h"
#include "descriptor_methods.h"
#include "compute_dish.h"

void bruteForceHammingSorted(cv::Mat img1, cv::Mat img2, Result res);
int bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res);
void bruteForceKNN(cv::Mat img1, foodTemplate food, cv::Mat dish,
                   cv::Mat &final, std::vector<FoodData> &boundingBoxes);
void checkType(cv::Mat &img1, cv::Mat &img2, Result &res);
void computeMinMaxCoordinates(cv::Mat &final, std::vector<cv::DMatch> &goodMatches, Result &res);

int bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res);
void bruteForceKNN(cv::Mat img1, foodTemplate food, cv::Mat dish, cv::Mat &final, std::vector<FoodData> &boundingBoxes,
                   int max_key, std::vector<Dish> &dishesData);

#endif // MATCHER_METHODS_H