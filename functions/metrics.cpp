#include "../headers/metrics.h"

using namespace std;

double calculateAveragePrecision(const std::vector<double> &results, const std::vector<double> &relevantDocs, double iouThreshold)
{
    int count = 0;
    double sumPrecision = 0.0;
    double sumIoU = 0.0;
    for (int i = 0; i < results.size(); i++)
    {
        if (find(relevantDocs.begin(), relevantDocs.end(), results[i]) != relevantDocs.end())
        {
            count++;
            sumPrecision += static_cast<double>(count) / (i + 1);

            // Calculate Intersection over Union (IoU)
            double intersection = count;
            double unionSize = i + 1;
            double iou = intersection / unionSize;
            sumIoU += iou;
        }
    }
    if (count == 0)
    {
        return 0.0;
    }
    double averageIoU = sumIoU / count;
    return sumPrecision / count;
}

void calculateMetrics(const std::vector<std::vector<double>> &allResults, const std::vector<std::vector<double>> &allRelevantDocs, double iouThreshold, double &map, double &meanIoU)
{
    double sumAP = 0.0;
    double sumIoU = 0.0;
    int numQueries = allResults.size();
    int validQueries = 0; // Count of queries with IoU threshold >= 0.5
    for (int i = 0; i < numQueries; ++i)
    {
        double ap = calculateAveragePrecision(allResults[i], allRelevantDocs[i], iouThreshold);
        if (iouThreshold >= 0.5)
        {
            sumAP += ap;
            sumIoU += meanIoU;
            validQueries++;
        }
    }
    if (validQueries == 0)
    {
        map = 0.0;
        meanIoU = 0.0;
    }
    else
    {
        map = sumAP / validQueries;
        meanIoU = sumIoU / validQueries;
    }
}
