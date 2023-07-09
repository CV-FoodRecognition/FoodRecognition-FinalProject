#ifndef LEFTOVER_H
#define LEFTOVER_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../headers/segmentation.h"
#include "../headers/descriptor_methods.h"
#include "../headers/matcher_methods.h"
#include "../headers/utils.h"
#include "../headers/ImageProcessor.h"

struct Couple
{
    cv::Mat original;
    cv::Mat leftover;
};

// funzione gestore
void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers,
                      const std::vector<int> &radia1, const std::vector<int> &radia2);

// uses all 3 measurments methods
void jointPredictions(std::vector<Couple> minDists,
                      std::vector<Couple> pairArea,
                      std::vector<Couple> pairMatches);

// ------------------------------------------------------------------------------------------------------ //
// measurments methods

std::vector<Couple> coupleSegmentColors(std::vector<cv::Mat> &originals, std::vector<cv::Mat> &leftovers);

std::vector<Couple> coupleClosestElements(const std::vector<double> &circleAreasOriginal,
                                          const std::vector<double> &circleAreasLeftover,
                                          const std::vector<cv::Mat> &originals,
                                          const std::vector<cv::Mat> &leftovers);

Couple coupleMaxMatches(const std::vector<int> &matches,
                        std::vector<cv::Mat> &leftovers, const cv::Mat &original);

std::vector<Couple> coupleMinAverageColor(const std::vector<cv::Scalar> &avgOriginals,
                                          const std::vector<cv::Scalar> &avgLefts,
                                          const std::vector<cv::Mat> &originals,
                                          const std::vector<cv::Mat> &leftovers);

// ---------------------------------------------------------------------------------------------------- //
// utils
bool checkCouplesEqual(const Couple &a, const Couple &b);

#endif // LEFTOVER_H