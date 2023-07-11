#ifndef LEFTOVER_CLASS_H
#define LEFTOVER_CLASS_H

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "../headers/segmentation.h"
#include "../headers/descriptor_methods.h"
#include "../headers/matcher_methods.h"
#include "../headers/utils.h"
#include "../headers/ImageProcessor.h"

class Leftover
{

public:
    // funzione gestore
    void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers,
                          const std::vector<int> &radia1, const std::vector<int> &radia2);

    // GETTERS
    std::vector<double> getCircleAreasOriginal() const { return circleAreasOriginal; }
    std::vector<double> getCircleAreasLeftover() const { return circleAreasLeftover; }
    std::vector<cv::Scalar> getAvgOriginals() const { return avgOriginals; }
    std::vector<cv::Scalar> getAvgLefts() const { return avgLefts; }
    std::vector<Couple> getMinDists() const { return minDists; }
    std::vector<Couple> getPairArea() const { return pairArea; }
    std::vector<Couple> getPairMatches() const { return pairMatches; }
    std::vector<Couple> getPairSegments() const { return pairSegments; }

private:
    std::vector<double> circleAreasOriginal; // areas of dishes in original tray
    std::vector<double> circleAreasLeftover; // areas of dishes in leftover tray
    std::vector<cv::Scalar> avgOriginals;    // vector of average colors for original images in the tray
    std::vector<cv::Scalar> avgLefts;        // vector of average colors for leftover images in the tray
    std::vector<Couple> minDists, pairArea, pairMatches, pairSegments;

    // ------------------------------------------------------------------------------------------------------ //
    // measurments methods

    std::vector<Couple> coupleSegmentColors(std::vector<cv::Mat> &originals, std::vector<cv::Mat> &leftovers);
    std::vector<Couple> coupleClosestElements(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);
    Couple coupleMaxMatches(const std::vector<int> &matches, std::vector<cv::Mat> &leftovers, const cv::Mat &original);
    std::vector<Couple> coupleMinAverageColor(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers);

    // ------------------------------------------------------------------------------------------------------ //
    // combines all measurments methods
    void jointPredictions();

    // ---------------------------------------------------------------------------------------------------- //
    // utils
    bool checkCouplesEqual(const Couple &a, const Couple &b);
    void printVector(const std::vector<Couple> &pairs, const std::string &title);
};

#endif // LEFTOVER_CLASS_H