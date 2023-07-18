#ifndef DETECT_CLASS_H
#define DETECT_CLASS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "utils.h"
#include "segmentation.h"
#include "descriptor_methods.h"
#include "matcher_methods.h"

int computeBestDish(foodTemplate food, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches);
cv::Scalar computeAvgColor(cv::Mat img);
cv::Scalar computeAvgColor(cv::Mat img, int id);
cv::Rect computeBox(cv::Mat &final, cv::Mat &dish);
void boundPasta(cv::Mat &dish, cv::Mat &final, std::string label, int id, std::vector<int> forbidden, int max_key, std::vector<FoodData> &foodData);
void boundPasta(cv::Mat &dish, cv::Mat &final, std::vector<std::string> labels, std::vector<int> ids, std::vector<int> forbidden,
                int max_key, std::vector<FoodData> &foodData);
void boundBread(cv::Mat &input, std::vector<cv::Mat> &dishes,
                cv::Mat &final, std::vector<FoodData> &foodData);
void drawBoundingBoxes(cv::Mat &final, std::vector<FoodData> &foodData);
void boundSalad(cv::Mat &input, std::vector<cv::Vec3f> accepted_circles,
                cv::Mat &final, std::vector<FoodData> &foodData);
void boundPotatoes(cv::Mat &dish, cv::Mat &final, std::vector<FoodData> &foodData, std::vector<int> forbidden, int max_key);
void detectAndCompute(cv::Mat in1, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches,
                      std::vector<cv::Vec3f> accepted_circles, std::vector<FoodData> &foodData, std::vector<foodTemplate> templates, cv::Mat &final);

#endif
