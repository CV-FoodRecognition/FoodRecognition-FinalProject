#include "../headers/leftover.h"
#include "../headers/utils.h"
#include "../headers/descriptor_methods.h"
#include "../headers/matcher_methods.h"

void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers,
                      const std::vector<int> &radia1, const std::vector<int> &radia2)
{
    Result res1, res2, res3;

    // leggo i 2 o 3 piatti originali
    cv::Mat original1 = removedDishes[0];
    cv::Mat original2 = removedDishes[1];
    cv::Mat original3;
    if (removedDishes.size() == 3)
        original3 = removedDishes[2];

    // Calcola avg color 1-3 piatti originali
    Scalar avgOriginal1 = computeAvgColor(original1);
    Scalar avgOriginal2 = computeAvgColor(original2);
    Scalar avgOriginal3;
    if (removedDishes.size() == 3)
        avgOriginal3 = computeAvgColor(original3);

    // Removes dishes from leftovers
    std::vector<cv::Mat> removedLeftovers;
    for (int d = 0; d < leftovers.size(); d++)
    {
        cv::Mat rmvDish = leftovers[d];
        removeDish(rmvDish);
        sharpenImg(rmvDish, SharpnessType::LAPLACIAN);
        removedLeftovers.push_back(rmvDish);
    }

    // Take leftovers
    cv::Mat left1 = removedLeftovers[0];
    cv::Mat left2 = removedLeftovers[1];
    cv::Mat left3;
    if (removedLeftovers.size() == 3)
        left3 = removedLeftovers[2];

    // Average color for leftovers
    Scalar avgLeft1 = computeAvgColor(left1);
    Scalar avgLeft2 = computeAvgColor(left2);
    Scalar avgLeft3;
    if (removedDishes.size() == 3)
        avgLeft3 = computeAvgColor(left3);

    // area cerchi
    std::vector<double> circleAreasOriginal;
    std::vector<double> circleAreasLeftover;
    // vector of matches
    std::vector<int> matches;

    std::vector<std::pair<cv::Mat, cv::Mat>> pairMatches;
    for (int i = 0; i < removedDishes.size(); i++)
    {
        res1 = useDescriptor(removedDishes[i], removedLeftovers[0], DescriptorType::SIFT);
        res2 = useDescriptor(removedDishes[i], removedLeftovers[1], DescriptorType::SIFT);

        int matches1 = bruteForceKNN(removedDishes[i], removedLeftovers[0], res1);
        int matches2 = bruteForceKNN(removedDishes[i], removedLeftovers[1], res2);
        matches.push_back(matches1);
        matches.push_back(matches2);

        if (removedDishes.size() == 3)
        {
            res3 = useDescriptor(removedDishes[i], removedLeftovers[2], DescriptorType::SIFT);
            int matches3 = bruteForceKNN(removedDishes[i], removedLeftovers[2], res3);
            matches.push_back(matches3);
        }

        // COUPLE by SIFT MATCHES
        std::pair<cv::Mat, cv::Mat> tempPair = computeMax(matches, leftovers, removedDishes[i]);
        pairMatches.push_back(tempPair);

        showImg("original", tempPair.first);
        showImg("leftover", tempPair.second);

        double circleOriginal = computeCircleArea(radia1[i]);
        double circleLeftover = computeCircleArea(radia2[i]);
        circleAreasOriginal.push_back(circleOriginal);
        circleAreasLeftover.push_back(circleLeftover);
    }

    // COUPLE by AREA CIRCLE
    std::vector<std::pair<cv::Mat, cv::Mat>> pairArea = coupleClosestElements(circleAreasOriginal, circleAreasLeftover,
                                                                              removedDishes, removedLeftovers);

    // COUPLE by AVERAGE COLOR
    std::vector<cv::Scalar> avgOriginals = {avgOriginal1, avgOriginal2};
    std::vector<cv::Scalar> avgLefts = {avgLeft1, avgLeft2};
    if (removedDishes.size() == 3)
    {
        avgOriginals.push_back(avgOriginal3);
        avgLefts.push_back(avgLeft3);
    }

    std::vector<std::pair<cv::Mat, cv::Mat>> minDists = minDistance(avgOriginals, avgLefts,
                                                                    removedDishes, removedLeftovers);

    // check if 3 coupling methods have given the same output pairs
    // if all 2 or 3 have given the same output pair, return it.
    // if all 3 methods have given 3 different pairs, pick random pair.
    jointPredictions(minDists, pairArea, pairMatches);
}

void jointPredictions(std::vector<std::pair<cv::Mat, cv::Mat>> minDists,
                      std::vector<std::pair<cv::Mat, cv::Mat>> pairArea,
                      std::vector<std::pair<cv::Mat, cv::Mat>> pairMatches)
{
    // check if 3 coupling methods have given the same output pairs
    // if all 2 or 3 have given the same output pair, return it.
    if (checkPairsEqual(pairMatches[0], minDists[0]) && checkPairsEqual(pairMatches[0], pairArea[0]))
    {
        showImg("original", pairMatches[0].first);
        showImg("leftover", pairMatches[0].second);
    }
    // if two methods have given the same output pair, return it.
    else if (checkPairsEqual(pairMatches[0], minDists[0]) || checkPairsEqual(pairMatches[0], pairArea[0]) ||
             checkPairsEqual(pairArea[0], minDists[0]))
    {
        showImg("original", pairMatches[0].first);
        showImg("leftover", pairMatches[0].second);
    }
    // if all 3 methods have given 3 different pairs, pick random pair.
    else
    {
        std::pair<cv::Mat, cv::Mat> tempPair = pairMatches[rand() % 3];
        showImg("original", tempPair.first);
        showImg("leftover", tempPair.second);
    }
}

std::pair<cv::Mat, cv::Mat> computeMax(const std::vector<int> &matches, const std::vector<cv::Mat> &leftovers, const cv::Mat &original)
{
    int maxIndex = std::distance(matches.begin(), std::max_element(matches.begin(), matches.end()));
    return std::make_pair(original, leftovers[maxIndex]);
}

std::vector<std::pair<cv::Mat, cv::Mat>> coupleClosestElements(const std::vector<double> &circleAreasOriginal,
                                                               const std::vector<double> &circleAreasLeftover,
                                                               const std::vector<cv::Mat> &originals,
                                                               const std::vector<cv::Mat> &leftovers)
{
    std::vector<std::pair<cv::Mat, cv::Mat>> result;
    for (int i = 0; i < circleAreasOriginal.size(); i++)
    {
        double original = circleAreasOriginal[i];
        double minDistance = std::numeric_limits<double>::max();
        int closestIndex = 0;
        for (int j = 0; j < circleAreasLeftover.size(); j++)
        {
            double leftover = circleAreasLeftover[j];
            double distance = std::abs(original - leftover);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestIndex = j;
            }
        }
        result.push_back(std::make_pair(originals[i], leftovers[closestIndex]));
    }
    return result;
}

std::vector<std::pair<cv::Mat, cv::Mat>> minDistance(const std::vector<cv::Scalar> &avgOriginals,
                                                     const std::vector<cv::Scalar> &avgLefts,
                                                     const std::vector<cv::Mat> &originals,
                                                     const std::vector<cv::Mat> &leftovers)
{
    std::vector<std::pair<cv::Mat, cv::Mat>> result;
    for (int i = 0; i < avgOriginals.size(); i++)
    {
        const cv::Scalar &avgOriginal = avgOriginals[i];
        double minDist = std::numeric_limits<double>::max();
        int closestIndex = 0;
        for (int j = 0; j < avgLefts.size(); j++)
        {
            const cv::Scalar &avgLeft = avgLefts[j];
            double dist = cv::norm(avgOriginal - avgLeft);
            if (dist < minDist)
            {
                minDist = dist;
                closestIndex = j;
            }
        }
        result.push_back(std::make_pair(originals[i], leftovers[closestIndex]));
    }
    return result;
}

bool checkPairsEqual(const std::pair<cv::Mat, cv::Mat> &a, const std::pair<cv::Mat, cv::Mat> &b)
{
    return cv::countNonZero(a.first != b.first) == 0 && cv::countNonZero(a.second != b.second) == 0;
}
