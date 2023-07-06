#ifndef METRICS_H
#define METRICS_H

#include <iostream>
#include <vector>

double calculateAveragePrecision(const std::vector<double> &results,
                                 const std::vector<double> &relevantDocs,
                                 double iouThreshold);
void calculateMetrics(const std::vector<std::vector<double>> &allResults,
                      const std::vector<std::vector<double>> &allRelevantDocs,
                      double iouThreshold, double &map, double &meanIoU);

#endif // METRICS_H