#include "../headers/metrics.h"

using namespace std;

void handlerFunction(const std::vector<Couple> &finalPairs, std::vector<Couple> &gtPairs)
{
    for (int i = 0; i < 8; i++)
    {
        std::string subDir = "Tray" + to_string(i + 1);
        for (int j = 0; j < 3; j++)
            readTray(gtPairs, subDir, j);
    }
}

void readTray(std::vector<Couple> &gtPairs, std::string subDir, int j)
{
    std::string directory = "../masks/" + subDir;
    std::string original = "food_image_mask.png";
    std::string leftover = "leftover" + to_string(j) + ".png";

    cv::Mat origImg = cv::imread(directory + original, CV_8UC1);
    cv::Mat leftImg = cv::imread(directory + leftover, CV_8UC1);

    std::vector<FoodData> dataOrig = parseGroundTruth("../bounding_boxes/" + subDir + "food_image_bounding_box.txt");
    std::vector<FoodData> dataLeft = parseGroundTruth("../bounding_boxes/" + subDir + "leftover1_bounding_box.txt");

    /*
        !! DISCLAIMER !!
        !! Da rivedere le strutture usate... Servono dei FoodContainer non dei FoodData !!
    */
    Couple c;
    c.leftoverBB = dataLeft;
    c.originalBB = dataOrig;
    gtPairs.push_back(c);
}

// Function that extract the ID and the coordinates
std::vector<FoodData> parseGroundTruth(const std::string &filename)
{
    std::vector<FoodData> groundTruth;

    std::ifstream inputFile(filename);
    if (!inputFile.is_open())
    {
        std::cout << "Failed to open file: " << filename << std::endl;
        return groundTruth;
    }

    std::string line;
    while (std::getline(inputFile, line))
    {
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
        while (std::getline(coordStream, coord, ','))
        {
            coords.push_back(std::stof(coord));
        }

        if (coords.size() != 4)
        {
            std::cout << "Invalid coordinate format: " << token << std::endl;
            continue;
        }

        FoodData bbox;
        bbox.ids[0] = id;
        bbox.box.x = coords[0];
        bbox.box.y = coords[1];
        bbox.box.width = coords[2];
        bbox.box.height = coords[3];

        groundTruth.push_back(bbox);
    }

    inputFile.close();

    return groundTruth;
}

// Function that returns the Intersection over Union for all the ground truths and all the predictions
std::vector<float> get_ious(const std::vector<FoodData> &ground_truths, const std::vector<FoodData> &preds)
{
    std::vector<float> ious;
    for (int i = 0; i < ground_truths.size() && i < preds.size(); i++)
    {
        float iou = get_iou(ground_truths[i], preds[i]);
        ious.push_back(iou);
    }
    return ious;
}

// For mAP 1
// Function that returns the Intersection over Union parameter
float get_iou(const FoodData &ground_truth, const FoodData &pred)
{
    float ix1 = std::max(ground_truth.box.x, pred.box.x); // Coordinates of the area of intersection
    float iy1 = std::max(ground_truth.box.y, pred.box.y);
    float ix2 = std::min(ground_truth.box.width, pred.box.width);
    float iy2 = std::min(ground_truth.box.height, pred.box.height);

    float i_height = std::max(iy2 - iy1 + 1, 0.0f); // Intersection height and width
    float i_width = std::max(ix2 - ix1 + 1, 0.0f);

    // INTERSECTION AREA
    float area_of_intersection = i_height * i_width;

    float gt_height = ground_truth.box.height - ground_truth.box.y + 1; // Ground Truth dimensions
    float gt_width = ground_truth.box.width - ground_truth.box.x + 1;

    float pred_height = pred.box.height - pred.box.y + 1; // Prediction dimensions
    float pred_width = pred.box.width - pred.box.x + 1;

    // UNION AREA
    float area_of_union = gt_height * gt_width + pred_height * pred_width - area_of_intersection;

    // RETURN IoU
    return area_of_intersection / area_of_union;
}

// For mAP 2
std::vector<std::string> getConfusionVector(const std::vector<float> &ious)
{
    std::vector<std::string> confusionVector;
    for (const auto &iou : ious)
    {
        if (iou == 0.0)
            confusionVector.push_back("FN");
        else if (iou <= IOUthresh)
            confusionVector.push_back("FP");
        else
            confusionVector.push_back("TP");
    }
    return confusionVector;
}

// For mAP 3
std::vector<int> getCumulativeTP(const std::vector<std::string> &matchStatuses)
{
    std::vector<int> cumulativeTP;
    int cumTP = 0;
    for (const auto &status : matchStatuses)
    {
        if (status == "TP")
            cumTP += 1;
        cumulativeTP.push_back(cumTP);
    }
    return cumulativeTP;
}

// For mAP 4
std::vector<int> getCumulativeFP(const std::vector<std::string> &matchStatuses)
{
    std::vector<int> cumulativeFP;
    int cumFP = 0;
    for (const auto &status : matchStatuses)
    {
        if (status == "FP")
            cumFP += 1;
        cumulativeFP.push_back(cumFP);
    }
    return cumulativeFP;
}

// For mAP 5
std::vector<float> getPrecision(const std::vector<int> &cumulativeTP, const std::vector<int> &cumulativeFP)
{
    std::vector<float> precision;
    for (int i = 0; i < cumulativeTP.size() && i < cumulativeFP.size(); i++)
    {
        float prec = cumulativeTP[i] / float(cumulativeTP[i] + cumulativeFP[i]);
        precision.push_back(prec);
    }
    return precision;
}

// For mAP 6
std::vector<float> getRecall(const std::vector<int> &cumulativeTP, float gtTotal)
{
    std::vector<float> recall;
    for (const auto &tp : cumulativeTP)
    {
        float rec = tp / gtTotal;
        recall.push_back(rec);
    }
    return recall;
}

// -------------------------------------------------------------------------------------------------------- //

// Function that returns the mean between all the ground truth bounding_boxes and the predicted one
float get_meaniou(std::vector<FoodDataContainer> &groundTruth, std::vector<FoodDataContainer> &predictions)
{
    for (auto &gt : groundTruth)
    {
        for (auto &pred : predictions)
        {
            if (gt.ids[0] == pred.ids[0])
            {
                float intersectionSegments = getIntersectionSegments(gt.segmentArea, pred.segmentArea).size();
                float unionSegments = getUnionSegments(gt.segmentArea, pred.segmentArea).size();

                if (intersectionSegments == 0 || unionSegments == 0)
                    return 0;

                else
                    return getIntersectionSegments(gt.segmentArea, pred.segmentArea).size() / getUnionSegments(gt.segmentArea, pred.segmentArea).size();
            }
            else
            {
                std::cerr << "Ids are not well predicted." << std::endl;

                float intersectionSegments = getIntersectionSegments(gt.segmentArea, pred.segmentArea).size();
                float unionSegments = getUnionSegments(gt.segmentArea, pred.segmentArea).size();

                if (intersectionSegments == 0 || unionSegments == 0)
                    return 0;

                else
                    return getIntersectionSegments(gt.segmentArea, pred.segmentArea).size() / getUnionSegments(gt.segmentArea, pred.segmentArea).size();
            }
        }
    }
}

std::set<cv::Point> getUnionSegments(cv::Mat &S1, cv::Mat &S2)
{
    std::set<cv::Point> unionSet;
    for (int i = 0; i < S1.rows; i++)
    {
        for (int j = 0; j < S1.cols; j++)
        {
            if (S1.at<uchar>(i, j) > 0)
                unionSet.insert(cv::Point(j, i));
        }
    }
    for (int i = 0; i < S2.rows; i++)
    {
        for (int j = 0; j < S2.cols; j++)
        {
            if (S2.at<uchar>(i, j) > 0)
                unionSet.insert(cv::Point(j, i));
        }
    }
    return unionSet;
}

std::set<cv::Point> getIntersectionSegments(cv::Mat &S1, cv::Mat &S2)
{
    std::set<cv::Point> intersectionSet;
    for (int i = 0; i < S1.rows; i++)
    {
        for (int j = 0; j < S1.cols; j++)
        {
            if (S1.at<uchar>(i, j) > 0 && S2.at<uchar>(i, j) > 0)
                intersectionSet.insert(cv::Point(j, i));
        }
    }
    return intersectionSet;
}

float calculatePixelRatio(int pixelsAfterimg, int pixelsBeforeimg)
{
    if (pixelsBeforeimg == 0)
    {
        std::cout << "Error: Division by zero." << std::endl;
        return 0;
    }
    return float(pixelsAfterimg) / pixelsBeforeimg;
}

/*
    @params:
        finalPairs: vector of Couples that contain all the infos about segmentation, areas,
                    and original and leftover matching
        gtMasks:    vector of Couples of all the areas in the grond truth masks

        !! DISCLAIMER !!
        !! There is the need to create a vector that only contains the segments of the gt masks !!
        !! This must be created in the Handler function of the metrics !!
*/
void computeBeforeAfterRatio(const std::vector<Couple> &finalPairs, const std::vector<Couple> &gtMasks)
{
    int pixelsAfterimg = 0, pixelsBeforeimg = 0;

    std::vector<float> ourRatios, gtRatios;

    // 1st PART: Our vector of Matches Originals / Leftovers //
    for (const Couple &c : finalPairs)
    {
        float areaLeftover, areaOriginal, result;

        // AREA LEFTOVER
        if (c.leftoverBB.areas.size() > 1)
            areaLeftover = c.leftoverBB.areas[0] + c.leftoverBB.areas[1]; // If there are more segments, it sums
                                                                          // yellow and blue segments (k=3)
        else                                                              // Else it only takes the yellow area
            areaLeftover = c.leftoverBB.areas[0];
        // AREA ORIGINAL
        if (c.originalBB.areas.size() > 1)
            areaOriginal = c.originalBB.areas[0] + c.originalBB.areas[1]; // If there are more segments, it sums
                                                                          // yellow and blue segments (k=3)
        else                                                              // Else it only takes the yellow area
            areaOriginal = c.originalBB.areas[0];

        // RESULT COMPUTATION
        if (areaOriginal == 0)
        {
            std::cerr << "Warning! Dividing by 0! \n";
            // ourRatios.push_back(0);
        }
        else
        {
            result = calculatePixelRatio(areaLeftover, areaOriginal);
            ourRatios.push_back(result);
        }
    }

    // 2nd PART: GT vector of Matches Originals / Leftovers //

    for (const Couple &c : gtMasks)
    {
        float areaLeftover, areaOriginal, result;
        if (c.leftoverBB.areas.size() > 1)
            areaLeftover = c.leftoverBB.areas[0] + c.leftoverBB.areas[1];
        else
            areaLeftover = c.leftoverBB.areas[0];

        if (c.originalBB.areas.size() > 1)
            areaOriginal = c.originalBB.areas[0] + c.originalBB.areas[1];
        else
            areaOriginal = c.originalBB.areas[0];

        if (areaOriginal == 0)
        {
            std::cerr << "Warning! Dividing by 0! \n";
            // gtRatios.push_back(0);
        }
        else
        {
            result = calculatePixelRatio(areaLeftover, areaOriginal);
            gtRatios.push_back(result);
        }
    }
}
