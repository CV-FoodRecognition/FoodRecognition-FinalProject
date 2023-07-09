#ifndef LEFTOVER_H
#define LEFTOVER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../headers/segmentation.h"
#include "../headers/utils.h"

// funzione gestore
void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers,
                      const std::vector<int> &radia1, const std::vector<int> &radia2);

// uses all 3 measurments methods
void jointPredictions(std::vector<std::pair<cv::Mat, cv::Mat>> minDists,
                      std::vector<std::pair<cv::Mat, cv::Mat>> pairArea,
                      std::vector<std::pair<cv::Mat, cv::Mat>> pairMatches);

// measurments methods
std::vector<std::pair<cv::Mat, cv::Mat>> coupleClosestElements(const std::vector<double> &circleAreasOriginal,
                                                               const std::vector<double> &circleAreasLeftover,
                                                               const std::vector<cv::Mat> &originals,
                                                               const std::vector<cv::Mat> &leftovers);

std::pair<cv::Mat, cv::Mat> computeMax(const std::vector<int> &matches,
                                       const std::vector<cv::Mat> &leftovers, const cv::Mat &original);

std::vector<std::pair<cv::Mat, cv::Mat>> minDistance(const std::vector<cv::Scalar> &avgOriginals,
                                                     const std::vector<cv::Scalar> &avgLefts,
                                                     const std::vector<cv::Mat> &originals,
                                                     const std::vector<cv::Mat> &leftovers);

// utils
bool checkPairsEqual(const std::pair<cv::Mat, cv::Mat> &a, const std::pair<cv::Mat, cv::Mat> &b);

#endif // LEFTOVER_H