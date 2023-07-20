#ifndef LEFTOVER_CLASS_H
#define LEFTOVER_CLASS_H

/*
Class written by @nicolacalzone
*/

#include <string>
#include <vector>
#include <limits>
#include <opencv2/opencv.hpp>
#include "segmentation.h"
#include "descriptor_methods.h"
#include "matcher_methods.h"
#include "utils.h"
#include "ImageProcessor.h"
#include "compute_dish.h"
#include "Dish.h"

class Leftover
{

public:
    // HANDLER
    std::vector<std::vector<SegmentCouple>> matchLeftovers(std::vector<cv::Mat> &removedDishes, std::vector<Dish> dishesData, const std::vector<cv::Mat> &leftovers,
                                                           cv::Mat leftover, const std::vector<int> &radia1, const std::vector<int> &radia2,
                                                           std::vector<FoodData> boxes);

    // GETTERS
    std::vector<Couple> getPairAvgColors() const { return pairAvgColors; }
    std::vector<Couple> getPairArea() const { return pairArea; }
    std::vector<Couple> getPairMatches() const { return pairMatches; }
    std::vector<Couple> getPairCieAvgs() const { return pairCieAvgs; }
    std::vector<cv::Mat> getLeftoverDishes() const { return leftoverDishes; }
    std::vector<cv::Mat> getSegmentedOriginals() const { return segmentedOriginals; }
    std::vector<cv::Mat> getSegmentedLeftovers() const { return segmentedLeftovers; }

private:
    // ------------------------------------------------------------------------------------------------------ //
    // Variables
    std::vector<Couple> pairAvgColors, pairArea, pairMatches, pairCieAvgs; // Final pair vectors.

    std::vector<cv::Mat> originalDishes, leftoverDishes, // original dishes and leftovers
        segmentedOriginals, segmentedLeftovers,          // segmented original and leftovers
        originalsCIELAB, leftoversCIELAB;                // vectors of CIELAB images for original and leftover images in the tray

    std::vector<cv::Scalar> avgCIELABOriginals; // vector of average colors for CIELAB original images in the tray
    std::vector<cv::Scalar> avgCIELABLefts;     // vector of average colors for CIELAB leftover images in the tray

    std::vector<cv::Scalar> avgOriginals; // vector of average colors for original images in the tray
    std::vector<cv::Scalar> avgLefts;     // vector of average colors for leftover images in the tray

    std::vector<double> circleAreasOriginal; // vector of areas of dishes in original tray
    std::vector<double> circleAreasLeftover; // vector of areas of dishes in leftover tray

    // ------------------------------------------------------------------------------------------------------ //
    // measurments methods      -->     @return: Couple

    std::vector<Couple> coupleCIELABColors(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers, bool flag);
    std::vector<Couple> coupleClosestElements(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);
    std::vector<Couple> coupleMinAverageColor(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);
    Couple coupleMaxMatches(const std::vector<int> &matches, std::vector<cv::Mat> &leftovers, const cv::Mat &original);

    // TODO:
    void segment(const std::vector<cv::Mat> &dishes, const std::vector<cv::Mat> &leftovers);

    // TODO:
    //  Bread segmenter and area computing
    void breadFinder(cv::Mat &leftover);

    // ------------------------------------------------------------------------------------------------------ //

    // combines all measurments methods
    std::vector<Couple> jointPredictions();
    // measurement methods for normal condition leftovers: 2:2 or 3:3 leftover-original
    void normalConditionsPrediction(std::vector<Couple> &finalPairs);
    // measurement methods for abnormal condition leftovers: 2:1, 3:1, 3:2 leftover-original
    void moreOriginalLessLeftovers(int type, std::vector<Couple> &finalPairs, std::vector<cv::Mat> &alreadyAssigned);
    // assign bounding boxes to every food
    SegmentCouple createCouple(Couple c, Dish orig, std::vector<SegmentCouple> &finalVec);
    std::vector<SegmentCouple> createFinalPairs(const Dish &dish, const std::vector<Couple> &finalPairs);
};

// ---------------------------------------------------------------------------------------------------- //
// UTILS

// utils for Leftover -- checks if two couples are the same couple
bool checkCouplesEqual(const Couple &a, const Couple &b);
// utils for Leftover -- checks if two Mat objects are the same
bool checkImageEqual(const cv::Mat &a, const cv::Mat &b);
// utils for Leftover -- prints a vector of couples (which can be all passed pairs: by matches, avgcolor, cielab_avgcolor etc.)
void printVector(const std::vector<Couple> &pairs, const std::string &title);
// utils for Leftover -- prints a vector of couples but prints their segmentation
void printVectorUpdate(const std::vector<SegmentCouple> &pairs, const std::string &title);
// utils for Leftover -- delta E computation for CIELAB comparison
double computeDeltaE(const cv::Scalar &c1, const cv::Scalar &c2);

#endif // LEFTOVER_CLASS_H