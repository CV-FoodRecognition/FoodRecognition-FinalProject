#ifndef COMPUTE_DISH_H
#define COMPUTE_DISH_H

#include <opencv2/core.hpp>
#include "../headers/utils.h"
#include "../headers/descriptor_methods.h"
#include "../headers/matcher_methods.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "../headers/Dish.h"
#include "../headers/utils.h"

cv::Scalar computeAvgColorHSV(cv::Mat img);
void drawBoundingBoxes(cv::Mat &final, std::vector<FoodData> &foodData);
cv::Mat getCorrectSegment(cv::Mat &dish, foodTemplate &foodTemplate, cv::Mat &maskYellow, cv::Mat &maskBlue);
cv::Rect computeBox(cv::Mat &dish);

int computeBestDish(foodTemplate food, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches);
cv::Scalar computeAvgColor(cv::Mat img);
cv::Scalar computeAvgColor(cv::Mat img, int id);
void boundPasta(cv::Mat &dish, cv::Mat &final, std::string label, int id, std::vector<int> forbidden,
                int max_key, std::vector<FoodData> &foodData);
cv::Rect computeBox(cv::Mat &final, cv::Mat &dish);
void boundBread(cv::Mat &input, std::vector<cv::Mat> &dishes,
                cv::Mat &final, std::vector<FoodData> &foodData);
void boundSalad(cv::Mat &input, std::vector<cv::Vec3f> accepted_circles,
                cv::Mat &final, std::vector<FoodData> &foodData,
                std::vector<cv::Mat> &dishes, std::vector<Dish> &dishesData);
void boundPotatoes(cv::Mat &dish, cv::Mat &final, std::vector<FoodData> &foodData, std::vector<int> forbidden,
                   int max_key, std::vector<Dish> &dishesData);
void detectAndCompute(cv::Mat in1, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches,
                      std::vector<cv::Vec3f> accepted_circles, std::vector<FoodData> &foodData,
                      std::vector<foodTemplate> templates, cv::Mat &final, std::vector<Dish> &dishesData);

#endif