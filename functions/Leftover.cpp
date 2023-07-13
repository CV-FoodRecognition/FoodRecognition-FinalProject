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

    originalDishes = removedDishes;
    leftoverDishes = removedLeftovers;

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
    std::cout << "c1" << std::endl;

    // COUPLE by AVERAGE COLOR
    pairAvgColors = coupleMinAverageColor(removedDishes, removedLeftovers);
    std::cout << "c2" << std::endl;

    // COUPLE by SEGMENT COLORS
    pairSegments = coupleSegmentColors(removedDishes, removedLeftovers);
    std::cout << "c3" << std::endl;

    /* printVector(pairArea, "Pair Area");
     printVector(pairAvgColors, "Pair Color");
     printVector(pairMatches, "Pair Matches");
     printVector(pairSegments, "Pair Segments"); */

    jointPredictions();
}

/*
    jointPredictions() function takes all the output couples from:
        - dish area comparison
        - segmented dish area comparison
        - number of matches applying SIFT
        - average color distance
    Then it checks if:
        - all four couples are the same     --> @returns: 100%  predicted couple
        - three couples are the same        --> @returns: 75%   predicted couple
        - two couples are the same
          and two couples are different     --> @returns: 50%   predicted couple
        - two couples are the same
          and two couples are the same      --> @returns: the pair with the most similar average color
        - no couples are the same           --> @returns: a random predicted couple
*/
void Leftover::jointPredictions()
{
    std::vector<Couple> finalPairs;
    std::vector<int> counterVec;

    // FOR EVERY ORIGINAL DISH     -->     4 PREDICTED LEFTOVERS
    // check how many predictions are the same
    for (int i = 0; i < pairMatches.size(); i++)
    {
        // initialize counter for equal results
        int counterEquals = 0;
        int m1 = 0, m2 = 0, m3 = 0;

        // Count number of same results from all measurement methods
        if (checkCouplesEqual(pairMatches[i], pairAvgColors[i]))
            counterEquals += 1;
        if (checkCouplesEqual(pairMatches[i], pairArea[i]))
            counterEquals += 1;
        if (checkCouplesEqual(pairMatches[i], pairSegments[i]))
            counterEquals += 1;

        // if all four couple are the same
        if (counterEquals == 3)
            finalPairs.push_back(pairMatches[i]);

        // if three couples are the same
        else if (counterEquals == 2)
            finalPairs.push_back(pairMatches[i]);

        // if two couples are the same and two different between each others
        else if (counterEquals == 1)
        {
            // The other two couples are equal but different from pairMatches
            // There is 50% chance for both pairMatches[i] and two of the other metrics.
            if (checkCouplesEqual(pairAvgColors[i], pairArea[i]) || checkCouplesEqual(pairAvgColors[i], pairSegments[i]) || checkCouplesEqual(pairArea[i], pairSegments[i]))
            {
                if (checkCouplesEqual(pairAvgColors[i], pairArea[i]) || checkCouplesEqual(pairAvgColors[i], pairSegments[i]))
                    finalPairs.push_back(pairAvgColors[i]);
                else
                    finalPairs.push_back(pairSegments[i]);
            }
            // The other two couples are equal but different between each others
            // it is still 50% chance that the predictions are good
            else
                finalPairs.push_back(pairMatches[i]);
        }
        // if all couples are different -- Count == 0
        else
            finalPairs.push_back(pairAvgColors[i]);

        counterVec.push_back(counterEquals);
    }

    printVector(finalPairs, "Final Predictions");
}

// ------------------------------------------------------------------------------------------------------ //
/*
    Computes Original and Leftover with similar yellow area (computing kmeans with 2 segments -
    where one is the background, black, one is the food in the dish, fixed to yellow)
    @returns: vector of Couple of Original,Leftover images with similar areas in the dish

    This metric is not very accurate because sometimes there might be no food left in the dish,
    so the area will differ a lot.
    But it can be used for leftover level 1, where the area stays similar.
*/
std::vector<Couple> Leftover::coupleSegmentColors(std::vector<cv::Mat> &originals, std::vector<cv::Mat> &leftovers)
{
    ImageProcessor ip;
    std::vector<SegmentAreas> originalSegmentAreas;
    for (int i = 0; i < originals.size(); i++)
    {
        cv::Mat segmentedOriginal = ip.kmeansSegmentation(2, originals[i]);
        SegmentAreas sa;
        sa.p1 = segmentedOriginal;
        computeSegmentArea(sa);
        originalSegmentAreas.push_back(sa);
    }

    std::vector<SegmentAreas> leftoverSegmentAreas;
    for (int i = 0; i < leftovers.size(); i++)
    {
        cv::Mat segmentedLeftover = ip.kmeansSegmentation(2, leftovers[i]);
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
            double distance = std::abs(originalSegmentAreas[i].areaYellow - leftoverSegmentAreas[j].areaYellow);
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
    Computes Original and Leftover with most matches
    @returns: Pair of Original,Leftover image with the least difference in area of circle
*/
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
    @returns: vector of Couple of Original,Leftover image with the least difference in area of circle
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
        std::cout << "before cycle" << std::endl;

        for (int j = 0; j < circleAreasLeftover.size() - 1; j++)
        {
            std::cout << "before in" << std::endl;
            double leftover = circleAreasLeftover[j];
            std::cout << "after in" << std::endl;
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
    @returns: vector of Couple of Original,Leftover image with the least distance of mean average color
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
        couple.dist = minDist;
        result.push_back(couple);
    }
    return result;
}

// ------------------------------------------------------------------------------------------------------ //

/*
    Checks if the Original is the same as the Original predicted
    and if the Leftover is the same as the Leftover predicted
*/
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
