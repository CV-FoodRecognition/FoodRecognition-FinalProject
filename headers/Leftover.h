#ifndef LEFTOVER_CLASS_H
#define LEFTOVER_CLASS_H

#include <string>
#include <vector>
#include <limits>
#include <opencv2/opencv.hpp>
#include "segmentation.h"
#include "descriptor_methods.h"
#include "matcher_methods.h"
#include "utils.h"
#include "ImageProcessor.h"

class Leftover
{

public:
    // HANDLER
    void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers,
                          const std::vector<int> &radia1, const std::vector<int> &radia2);

    // GETTERS
    std::vector<Couple> getPairAvgColors() const { return pairAvgColors; }
    std::vector<Couple> getPairArea() const { return pairArea; }
    std::vector<Couple> getPairMatches() const { return pairMatches; }
    std::vector<Couple> getPairCieAvgs() const { return pairCieAvgs; }
    std::vector<cv::Mat> getLeftoverDishes() const { return leftoverDishes; }
    std::vector<cv::Mat> getSegmentedOriginals() const { return segmentedOriginal; }
    std::vector<cv::Mat> getSegmentedLeftovers() const { return segmentedLeftovers; }

private:
    // ------------------------------------------------------------------------------------------------------ //
    // Variables
    std::vector<double> circleAreasOriginal;    // vector of areas of dishes in original tray
    std::vector<double> circleAreasLeftover;    // vector of areas of dishes in leftover tray
    std::vector<cv::Scalar> avgOriginals;       // vector of average colors for original images in the tray
    std::vector<cv::Scalar> avgLefts;           // vector of average colors for leftover images in the tray
    std::vector<cv::Mat> originalsCIELAB;       // vector of CIELAB images for original images in the tray
    std::vector<cv::Mat> leftoversCIELAB;       // vector of CIELAB images for leftover images in the tray
    std::vector<cv::Scalar> avgCIELABOriginals; // vector of average colors for CIELAB original images in the tray
    std::vector<cv::Scalar> avgCIELABLefts;     // vector of average colors for CIELAB leftover images in the tray

    std::vector<Couple> pairAvgColors, pairArea, pairMatches, pairCieAvgs;
    std::vector<cv::Mat> originalDishes, leftoverDishes, segmentedOriginal, segmentedLeftovers;

    // ------------------------------------------------------------------------------------------------------ //
    // measurments methods      -->     @return: Couple

    std::vector<Couple> coupleCIELABColors(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);
    std::vector<Couple> coupleClosestElements(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);
    std::vector<Couple> coupleMinAverageColor(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);
    Couple coupleMaxMatches(const std::vector<int> &matches, std::vector<cv::Mat> &leftovers, const cv::Mat &original);

    // TODO:
    // All dishes must be linked to different dishes, one leftover dish cannot be predicted for two original dishes
    void allDishesDifferent(std::vector<Couple> &finalPairs, const std::vector<int> &counterVec);

    // ------------------------------------------------------------------------------------------------------ //

    // combines all measurments methods
    void jointPredictions();
    // measurement methods for normal condition leftovers: 2:2 or 3:3 leftover-original
    void normalConditionsPrediction(std::vector<Couple> &finalPairs);
    // measurement methods for abnormal condition leftovers: 2:1, 3:1, 3:2 leftover-original
    void moreOriginalLessLeftovers(int type, std::vector<Couple> &finalPairs, std::vector<cv::Mat> &alreadyAssigned);
    // assign bounding boxes to every food
    void assignBoundingBoxes(std::vector<BoundingBox> &boxes, std::vector<BoundingBox> &results, std::vector<cv::Mat> &leftovers);
};

// ---------------------------------------------------------------------------------------------------- //
// UTILS

// utils for Leftover -- checks if two couples are the same couple
bool checkCouplesEqual(const Couple &a, const Couple &b);
// utils for Leftover -- prints a vector of couples (which can be all passed pairs: by matches, avgcolor, cielab_avgcolor etc.)
void printVector(const std::vector<Couple> &pairs, const std::string &title);

#endif // LEFTOVER_CLASS_H