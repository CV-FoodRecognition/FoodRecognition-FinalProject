#include "../headers/Leftover.h"

void Leftover::computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers,
                                const std::vector<int> &radia1, const std::vector<int> &radia2)
{
    Result res1, res2, res3;
    bool hasThreeOriginals = removedDishes.size() == 3;

    std::cout << "bool: " << hasThreeOriginals << std::endl;

    // leggo i 2 o 3 piatti originali
    cv::Mat original1 = removedDishes[0];
    cv::Mat original2 = removedDishes[1];

    // Calcola avg color 1 and 2 for piatti originali
    cv::Scalar avgOriginal1 = computeAvgColor(original1);
    cv::Scalar avgOriginal2 = computeAvgColor(original2);

    // Removes dishes from leftovers
    std::vector<cv::Mat> removedLeftovers;
    for (int d = 0; d < leftovers.size(); d++)
    {
        cv::Mat rmvDish = leftovers[d];
        removeDish(rmvDish);
        sharpenImg(rmvDish, SharpnessType::LAPLACIAN);
        removedLeftovers.push_back(rmvDish);
    }

    bool hasThreeLeftovers = removedLeftovers.size() == 3;

    std::cout << "original size " << removedDishes.size() << std::endl;
    std::cout << "leftover size " << removedLeftovers.size() << std::endl;

    // Take leftovers 1 and 2
    cv::Mat left1 = removedLeftovers[0];
    cv::Mat left2 = removedLeftovers[1];

    // Average color for leftovers 1 and 2
    cv::Scalar avgLeft1 = computeAvgColor(left1);
    cv::Scalar avgLeft2 = computeAvgColor(left2);

    std::cout << "average leftover colors " << std::endl;

    // Vectors of average colors for 1 and 2
    avgOriginals = {avgOriginal1, avgOriginal2};
    avgLefts = {avgLeft1, avgLeft2};

    // IF ORIGINAL DISHES ARE THREE...
    cv::Mat original3, left3;
    cv::Scalar avgOriginal3, avgLeft3;
    // 3 ORIGINALS
    if (hasThreeOriginals)
    {
        original3 = removedDishes[2];
        avgOriginal3 = computeAvgColor(original3);
        avgOriginals.push_back(avgOriginal3);
    }
    // 3 LEFTOVERS
    if (hasThreeLeftovers)
    {
        left3 = removedLeftovers[2];
        avgLeft3 = computeAvgColor(left3);
        avgLefts.push_back(avgLeft3);
    }

    std::vector<int> matches; // vector of matches
    /*
        For every circle in removedDishes (original dishes):
            - compute SIFT with the three leftover dishes with index 0,1,2
            - knn matching with the three leftover dishes
            - coupleMaxMatches() => see which leftover has given the most matches to the original dish
            - @returns coupleMaxMatches(): pair of two most matched Mat objects
            - compute area of Original Dish, compute area of Leftover picked by # of matches
            - add areas to vector of original areas and vector of leftover areas
    */
    for (int i = 0; i < removedDishes.size(); i++)
    {
        // Descriptor creation for SIFT
        res1 = useDescriptor(removedDishes[i], removedLeftovers[0], DescriptorType::SIFT);
        res2 = useDescriptor(removedDishes[i], removedLeftovers[1], DescriptorType::SIFT);
        // Brute Force KNN for SIFT descriptors
        int matches1 = bruteForceKNN(removedDishes[i], removedLeftovers[0], res1);
        int matches2 = bruteForceKNN(removedDishes[i], removedLeftovers[1], res2);
        // Matches stores the number of matches
        matches.push_back(matches1);
        matches.push_back(matches2);

        if (hasThreeOriginals && hasThreeLeftovers)
        {
            res3 = useDescriptor(removedDishes[i], removedLeftovers[2], DescriptorType::SIFT);
            int matches3 = bruteForceKNN(removedDishes[i], removedLeftovers[2], res3);
            matches.push_back(matches3);
        }

        // COUPLE by SIFT MATCHES
        Couple tempPair = coupleMaxMatches(matches, removedLeftovers, removedDishes[i]);
        pairMatches.push_back(tempPair);

        double circleOriginal = computeCircleArea(radia1[i]);
        double circleLeftover = computeCircleArea(radia2[i]);
        circleAreasOriginal.push_back(circleOriginal);
        circleAreasLeftover.push_back(circleLeftover);
    }

    std::cout << "lenght areas original: " << circleAreasOriginal.size() << std::endl;
    std::cout << "lenght areas leftovers: " << circleAreasLeftover.size() << std::endl;

    // COUPLE by AREA CIRCLE
    pairArea = coupleClosestElements(removedDishes, removedLeftovers);

    // COUPLE by AVERAGE COLOR
    minDists = coupleMinAverageColor(removedDishes, removedLeftovers);

    // COUPLE by SEGMENT COLORS
    pairSegments = coupleSegmentColors(removedDishes, removedLeftovers);

    printVector(pairArea, "Pair Area");
    printVector(minDists, "Pair Color");
    printVector(pairMatches, "Pair Matches");
    printVector(pairSegments, "Pair Segments");

    // jointPredictions(minDists, pairArea, pairMatches);
}

/*
    This function takes all the three output couples from:
        - area comparison
        - # of matches
        - average color distance
    Then it checks if:
        - all three couples are the same    --> @returns: 100% predicted couple
        - at least two couples are the same --> @returns: 66% majoritary couple
        - no couples are the same           --> @returns: a random prediction
*/
void Leftover::jointPredictions()
{
    // check if 3 coupling methods have given the same output pairs
    for (int i = 0; i < pairMatches.size(); i++)
    {
        // if all 2 or 3 have given the same output pair, return it.
        if (checkCouplesEqual(pairMatches[i], minDists[i]) && checkCouplesEqual(pairMatches[i], pairArea[i]))
        {
            showImg("original", pairMatches[i].original);
            showImg("leftover", pairMatches[i].leftover);
        }
    }

    for (int i = 0; i < pairMatches.size(); i++)
    {
        // if two methods have given the same output pair, return it.
        if (checkCouplesEqual(pairMatches[i], minDists[i]) || checkCouplesEqual(pairMatches[i], pairArea[i]) ||
            checkCouplesEqual(pairArea[i], minDists[i]))
        {
            showImg("original", pairMatches[i].original);
            showImg("leftover", pairMatches[i].leftover);
        }
    }

    // if all 3 methods have given 3 different pairs, pick random pair.
    std::vector<int> usedIndexes;
    for (int i = 0; i < pairMatches.size(); i++)
    {
        int index;
        do
        {
            index = rand() % pairMatches.size();
        } while (std::find(usedIndexes.begin(), usedIndexes.end(), index) != usedIndexes.end());
        usedIndexes.push_back(index);

        Couple tempPair = pairMatches[index];
        showImg("original", tempPair.original);
        showImg("leftover", tempPair.leftover);
    }
}

// ------------------------------------------------------------------------------------------------------ //

/*
    Computes Original and Leftover with similar yellow and blue area (computing kmeans with 2 segments)
    @returns: Pair of Original,Leftover images
*/
std::vector<Couple> Leftover::coupleSegmentColors(std::vector<cv::Mat> &originals, std::vector<cv::Mat> &leftovers)
{
    ImageProcessor ip;
    std::vector<SegmentAreas> originalSegmentAreas;
    for (int i = 0; i < originals.size(); i++)
    {
        cv::Mat segmentedOriginal = ip.kmeansSegmentation(3, originals[i]);
        SegmentAreas sa;
        sa.p1 = segmentedOriginal;
        computeSegmentArea(sa);
        originalSegmentAreas.push_back(sa);
    }

    std::vector<SegmentAreas> leftoverSegmentAreas;
    for (int i = 0; i < leftovers.size(); i++)
    {
        cv::Mat segmentedLeftover = ip.kmeansSegmentation(3, leftovers[i]);
        SegmentAreas sa;
        sa.p1 = segmentedLeftover;
        computeSegmentArea(sa);
        leftoverSegmentAreas.push_back(sa);
    }

    // Find best matching pairs according to segment areas
    std::vector<Couple> result;
    for (int i = 0; i < originalSegmentAreas.size(); i++)
    {
        double minDistance = std::numeric_limits<double>::max();
        int closestIndex = 0;
        for (int j = 0; j < leftoverSegmentAreas.size(); j++)
        {
            double distance = std::abs(originalSegmentAreas[i].areaYellow - leftoverSegmentAreas[j].areaYellow) +
                              std::abs(originalSegmentAreas[i].areaBlue - leftoverSegmentAreas[j].areaBlue);
            // std::abs(originalSegmentAreas[i].areaBlack - leftoverSegmentAreas[j].areaBlack);
            if (distance < minDistance)
            {
                minDistance = distance;
                closestIndex = j;
            }
        }
        Couple couple;
        couple.original = originals[i];
        couple.leftover = leftovers[closestIndex];
        result.push_back(couple);
    }
    return result;
}

Couple Leftover::coupleMaxMatches(const std::vector<int> &matches,
                                  std::vector<cv::Mat> &leftovers, const cv::Mat &original)
{
    if (matches[0] >= matches[1] && matches[0] >= matches[2])
    {
        Couple couple;
        couple.leftover = leftovers[0];
        couple.original = original;
        return couple;
    }
    else if (matches[1] >= matches[0] && matches[1] >= matches[2])
    {
        Couple couple;
        couple.leftover = leftovers[1];
        couple.original = original;
        return couple;
    }
    else
    {
        Couple couple;
        couple.leftover = leftovers[2];
        couple.original = original;
        return couple;
    }
}

/*
    Computes Original and Leftover with least difference in area of circle
    @returns: Pair of Original,Leftover image with the least difference in area of circle
*/
std::vector<Couple> Leftover::coupleClosestElements(const std::vector<cv::Mat> &originals,
                                                    const std::vector<cv::Mat> &leftovers)
{
    std::vector<Couple> result;
    for (int i = 0; i < circleAreasOriginal.size(); i++)
    {
        double original = circleAreasOriginal[i];
        double minDistance = 1000000;
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
        Couple couple;
        couple.original = originals[i];
        couple.leftover = leftovers[closestIndex];
        result.push_back(couple);
    }
    return result;
}

/*
    Computes minDistance between average colors of the images
    @returns: Pair of Original,Leftover image with the least distance of mean average color
*/
std::vector<Couple> Leftover::coupleMinAverageColor(const std::vector<cv::Mat> &originals,
                                                    const std::vector<cv::Mat> &leftovers)
{
    std::vector<Couple> result;
    for (int i = 0; i < avgOriginals.size(); i++)
    {
        const cv::Scalar &avgOriginal = avgOriginals[i];
        double minDist = 1000000;
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
        Couple couple;
        couple.original = originals[i];
        couple.leftover = leftovers[closestIndex];
        result.push_back(couple);
    }
    return result;
}

// ------------------------------------------------------------------------------------------------------ //

bool checkCouplesEqual(const Couple &a, const Couple &b)
{
    cv::Mat aOriginalGray, bOriginalGray, aLeftoverGray, bLeftoverGray;
    cv::cvtColor(a.original, aOriginalGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b.original, bOriginalGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(a.leftover, aLeftoverGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b.leftover, bLeftoverGray, cv::COLOR_BGR2GRAY);

    return cv::countNonZero(aOriginalGray != bOriginalGray) == 0 && cv::countNonZero(aLeftoverGray != bLeftoverGray) == 0;
}

void printVector(const std::vector<Couple> &pairs, const std::string &title)
{
    for (const auto &pair : pairs)
        concatShowImg(title, pair.original, pair.leftover);
}
