#ifndef METRICS_H
#define METRICS_H

#include <iostream>
#include <vector>

/* struct BoundingBox {
    int id;  //ID del cibo
    float x1, y1, x2, y2;  //Coordinate del rect
};*/

float get_iou(const FoodData &ground_truth, const FoodData &pred);

float get_meaniou(std::vector<FoodData> &groundTruth, std::vector<FoodData> &predictions);

double calculatePixelRatio(int pixelsAfterimg, int pixelsBeforeimg);

/*double calculateAveragePrecision(const std::vector<double> &results,
                                 const std::vector<double> &relevantDocs,
                                 double iouThreshold);
void calculateMetrics(const std::vector<std::vector<double>> &allResults,
                      const std::vector<std::vector<double>> &allRelevantDocs,
                      double iouThreshold, double &map, double &meanIoU);*/

#endif // METRICS_H