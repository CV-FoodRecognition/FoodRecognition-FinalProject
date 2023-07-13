#include "../headers/metrics.h"
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

struct BoundingBox {
    int id;  //ID del cibo
    float x1, y1, x2, y2;  //Coordinate del rect
};

std::vector<BoundingBox> parseGroundTruth(const std::string& filename) {
    std::vector<BoundingBox> groundTruth;

    std::ifstream inputFile(filename);
    if (!inputFile.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return groundTruth;
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        std::istringstream iss(line);
        std::string token;

        // Extract ID
        std::getline(iss, token, ':');
        int id = std::stoi(token);

        // Extract coordinates
        std::getline(iss, token, '[');
        std::getline(iss, token, ']');

        std::istringstream coordStream(token);
        std::vector<float> coords;
        std::string coord;
        while (std::getline(coordStream, coord, ',')) {
            coords.push_back(std::stof(coord));
        }

        if (coords.size() != 4) {
            std::cout << "Invalid coordinate format: " << token << std::endl;
            continue;
        }

        BoundingBox bbox;
        bbox.id = id;
        bbox.x1 = coords[0];
        bbox.y1 = coords[1];
        bbox.x2 = coords[2];
        bbox.y2 = coords[3];

        groundTruth.push_back(bbox);
    }

    inputFile.close();

    return groundTruth;
}


float get_iou(const BoundingBox& ground_truth, const BoundingBox& pred) {
    
    // Coordinates of the area of intersection
    float ix1 = std::max(ground_truth.x1, pred.x1);
    float iy1 = std::max(ground_truth.y1, pred.y1);
    float ix2 = std::min(ground_truth.x2, pred.x2);
    float iy2 = std::min(ground_truth.y2, pred.y2);

    // Intersection height and width
    float i_height = std::max(iy2 - iy1 + 1, 0.0f);
    float i_width = std::max(ix2 - ix1 + 1, 0.0f);

    float area_of_intersection = i_height * i_width;

    // Ground Truth dimensions
    float gt_height = ground_truth.y2 - ground_truth.y1 + 1;
    float gt_width = ground_truth.x2 - ground_truth.x1 + 1;

    // Prediction dimensions
    float pred_height = pred.y2 - pred.y1 + 1;
    float pred_width = pred.x2 - pred.x1 + 1;

    float area_of_union = gt_height * gt_width + pred_height * pred_width - area_of_intersection;

    float iou = area_of_intersection / area_of_union;

    return iou;
}

float get_meaniou(std::vector<BoundingBox>& groundTruth, std::vector<BoundingBox>& predictions)
{
    float sumIOU = 0.0;
    int numPairs = 0;

    for (const auto& gt : groundTruth) {
        for (const auto& pred : predictions) {
            if (gt.id == pred.id) {
                float iou = get_iou(gt, pred);
                sumIOU += iou;
                numPairs++;
            }
        }
    }

    if (numPairs > 0) {
        float meanIOU = sumIOU / numPairs;
        return meanIOU;
    }

}




/*double calculateAveragePrecision(const std::vector<double> &results, const std::vector<double> &relevantDocs, double iouThreshold)
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
}*/
