#ifndef LEFTOVER_H
#define LEFTOVER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../headers/segmentation.h"
#include "../headers/utils.h"

struct Couple {
        cv::Mat leftover;
        cv::Mat original;
};

//funzione gestore
void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers);

//first level
void firstLevel();

//second level
void secondLevel();

//third level
void thirdLevel();

Couple computeMax(int matches1, int matches2, int matches3, std::vector<cv::Mat> leftovers, const cv::Mat& original);
Couple computeMax(int matches1, int matches2, std::vector<cv::Mat> leftovers, const cv::Mat& original);


#endif // LEFTOVER_H