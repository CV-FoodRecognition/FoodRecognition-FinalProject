#include "../headers/Leftover.h"

/*
Class written by @nicolacalzone
*/

std::vector<std::vector<SegmentCouple>> Leftover::matchLeftovers(std::vector<cv::Mat> &removedDishes, std::vector<Dish> dishesData, const std::vector<cv::Mat> &leftovers,
                                                                 cv::Mat leftover, const std::vector<int> &radia1, const std::vector<int> &radia2,
                                                                 std::vector<FoodData> boxes)
{

    Result res1, res2, res3;
    bool hasThreeOriginals = removedDishes.size() == 3;

    std::cout << "hasThreeOriginals: " << hasThreeOriginals << std::endl;

    // Original FOR 1 DISH
    cv::Mat original1 = removedDishes[0];
    cv::Scalar avgOriginal1 = computeAvgColor(original1);

    cv::Mat original1CIE = convertBGRtoCIELAB(original1);
    cv::Scalar avgCIELABOriginal1 = computeAvgColorCIELAB(original1CIE);

    avgOriginals.push_back(avgOriginal1);
    avgCIELABOriginals.push_back(avgCIELABOriginal1);

    // Original FOR 2 DISHES
    if (removedDishes.size() > 1)
    {
        cv::Mat original2 = removedDishes[1];
        cv::Scalar avgOriginal2 = computeAvgColor(original2);

        cv::Mat original2CIE = convertBGRtoCIELAB(original2);
        cv::Scalar avgCIELABOriginal2 = computeAvgColorCIELAB(original2CIE);

        avgOriginals.push_back(avgOriginal2);
        avgCIELABOriginals.push_back(avgCIELABOriginal2);
    }

    // Removes dishes from leftovers
    std::vector<cv::Mat> removedLeftovers;
    for (int d = 0; d < leftovers.size(); d++)
    {
        cv::Mat rmvDish = leftovers[d];
        removeDish(rmvDish);
        sharpenImg(rmvDish, SharpnessType::LAPLACIAN);
        removedLeftovers.push_back(rmvDish);
    }

    bool hasThreeLeftovers = (removedLeftovers.size() == 3);
    std::cout << "hasThreeLeftovers: " << hasThreeLeftovers << std::endl;

    if (hasThreeLeftovers && !hasThreeOriginals)
    {
        std::cerr << "The leftover does not belong to the tray in input.";
        std::vector<std::vector<SegmentCouple>> empty;
        return empty;
    }

    // LEFT for 1 DISH
    cv::Mat left1 = removedLeftovers[0];          // removed dish
    cv::Scalar avgLeft1 = computeAvgColor(left1); // average color

    cv::Mat left1CIE = convertBGRtoCIELAB(left1);
    cv::Scalar avgCIELABLeft1 = computeAvgColorCIELAB(left1CIE); // average cielab color

    avgLefts.push_back(avgLeft1);
    avgCIELABLefts.push_back(avgCIELABLeft1);

    // LEFT for 2 DISHES
    if (removedLeftovers.size() > 1)
    {
        cv::Mat left2 = removedLeftovers[1];
        cv::Scalar avgLeft2 = computeAvgColor(left2);

        cv::Mat left2CIE = convertBGRtoCIELAB(left2);
        cv::Scalar avgCIELABLeft2 = computeAvgColorCIELAB(left2CIE);

        avgLefts.push_back(avgLeft2);
        avgCIELABLefts.push_back(avgCIELABLeft2);
    }

    // IF ORIGINAL DISHES ARE 3...
    cv::Mat original3, left3;
    cv::Scalar avgOriginal3, avgLeft3, avgCIELABOriginal3, avgCIELABLeftover3;
    if (hasThreeOriginals)
    { // 3 ORIGINALS
        original3 = removedDishes[2];
        avgOriginal3 = computeAvgColor(original3);

        cv::Mat original3CIE = convertBGRtoCIELAB(original3);
        cv::Scalar avgCIELABOriginal3 = computeAvgColorCIELAB(original3CIE);

        avgOriginals.push_back(avgOriginal3);
        avgCIELABOriginals.push_back(avgCIELABOriginal3);
    }
    // IF LEFTOVER DISHES ARE 3...
    if (hasThreeLeftovers)
    { // 3 LEFTOVERS
        left3 = removedLeftovers[2];
        avgLeft3 = computeAvgColor(left3);

        cv::Mat left3CIE = convertBGRtoCIELAB(left3);
        cv::Scalar avgCIELABLeft3 = computeAvgColorCIELAB(left3CIE);

        avgCIELABLeftover3 = computeAvgColorCIELAB(left3);
        avgLefts.push_back(avgLeft3);
        avgCIELABLefts.push_back(avgCIELABLeftover3);
    }

    originalDishes = removedDishes;
    leftoverDishes = removedLeftovers;

    // breadFinder(leftover);

    /*
        For every circle in removedDishes (original dishes):
            - compute SIFT with the three leftover dishes with index 0,1,2
            - knn matching with the three leftover dishes
            - coupleMaxMatches() => see which leftover has given the most matches to the original dish
            - @returns coupleMaxMatches(): pair of two most matched Mat objects
            - compute area of Original Dish, compute area of Leftover picked by # of matches
            - add areas to vector of original areas and vector of leftover areas
    */
    std::vector<int>
        matches; // vector of matches, is cleared at every iteration
    for (int i = 0; i < removedDishes.size(); i++)
    {
        res1 = useDescriptor(removedDishes[i], removedLeftovers[0], DescriptorType::SIFT);
        int matches1 = bruteForceKNN(removedDishes[i], removedLeftovers[0], res1);
        matches.push_back(matches1);

        // There are 2 leftovers or 3 leftovers
        if (removedLeftovers.size() > 1)
        {
            res2 = useDescriptor(removedDishes[i], removedLeftovers[1], DescriptorType::SIFT);
            int matches2 = bruteForceKNN(removedDishes[i], removedLeftovers[1], res2);
            matches.push_back(matches2);
        }

        // There are 3 leftovers
        if (removedLeftovers.size() > 2)
        {
            res3 = useDescriptor(removedDishes[i], removedLeftovers[2], DescriptorType::SIFT);
            int matches3 = bruteForceKNN(removedDishes[i], removedLeftovers[2], res3);
            matches.push_back(matches3);
        }

        Couple tempPair = coupleMaxMatches(matches, removedLeftovers, removedDishes[i]);
        pairMatches.push_back(tempPair);

        double circleOriginal = computeCircleArea(radia1[i]);
        circleAreasOriginal.push_back(circleOriginal);

        matches.clear(); // clear the dishes from previous matches of the dishes
    }

    for (int i = 0; i < removedLeftovers.size(); i++)
    {
        double circleLeftover = computeCircleArea(radia2[i]);
        circleAreasLeftover.push_back(circleLeftover);
    }

    std::cout << "pairMatches: " << pairMatches.size() << std::endl;

    // COUPLE by AREA CIRCLE
    pairArea = coupleClosestElements(removedDishes, removedLeftovers);
    std::cout << "pairArea: " << pairArea.size() << std::endl;

    // COUPLE by AVERAGE COLOR
    pairAvgColors = coupleMinAverageColor(removedDishes, removedLeftovers);
    std::cout << "pairAvgColors: " << pairAvgColors.size() << std::endl;

    // COUPLE by SEGMENT COLORS
    bool flag = false;
    pairCieAvgs = coupleCIELABColors(removedDishes, removedLeftovers, flag);
    std::cout << "pairAvgCIELAB: " << pairCieAvgs.size() << std::endl;

    // RESULTS OF ALL METHODOLOGIES
    /*printVector(pairArea, "Pair Area");
    printVector(pairAvgColors, "Pair Color");
    printVector(pairMatches, "Pair Matches");
    printVector(pairCieAvgs, "Pair CIE");*/

    std::vector<Couple> finalPairs = jointPredictions();
    // printVector(finalPairs, "Predictions");

    std::vector<std::vector<SegmentCouple>> output;

    for (Dish dish : dishesData)
    {
        std::vector<SegmentCouple> segmentedFinal = createFinalPairs(dish, finalPairs);
        std::cout << "Size Segmented Vector: " << segmentedFinal.size();
        printVectorUpdate(segmentedFinal, "Pair Segments for ID: ");
        output.push_back(segmentedFinal);
    }

    return output;
}

/*
    jointPredictions() function takes all the output couples from:
        - dish area comparison
        - average color CIELAB dish area comparison
        - number of matches applying SIFT
        - average color distance
    Then it checks if:
        - all four couples are the same     --> @returns: 100%  predicted couple
        - three couples are the same        --> @returns: 75%   predicted couple
        - two couples are the same
          and two couples are different     --> @returns: 50%   predicted couple
        - two couples are the same
          and two couples are the same      --> @returns: 50%   predicted couple with most similar average color
        - no couples are the same           --> @returns: a random couple
*/
std::vector<Couple> Leftover::jointPredictions()
{
    bool onlyOneLeftover = leftoverDishes.size() == 1;
    bool twoLeftoversTwoOriginals = (leftoverDishes.size() == 2 && originalDishes.size() == 2);
    bool threeLeftoversThreeOriginals = (leftoverDishes.size() == 3 && originalDishes.size() == 3);
    bool twoLeftoversThreeOriginals = leftoverDishes.size() == 2 && originalDishes.size() == 3;

    std::vector<Couple> finalPairs;
    std::vector<cv::Mat> alreadyAssigned;
    /*
        Initial Checks
        If leftover.size() == 1 --> all original dishes would predict the same leftover.
        So.. We need to order all the couples by number of matches.
    */
    if (onlyOneLeftover)
    {
        std::cout << "SELECTED PATH 1 -- 2/3 originals 1 leftover" << std::endl;
        int select = 1;
        moreOriginalLessLeftovers(select, finalPairs, alreadyAssigned);
    }
    else if (twoLeftoversThreeOriginals)
    {
        std::cout << "SELECTED PATH 2 -- 3 originals 2 leftovers" << std::endl;
        int select = 2;
        moreOriginalLessLeftovers(select, finalPairs, alreadyAssigned);
    }
    else if (twoLeftoversTwoOriginals || threeLeftoversThreeOriginals)
    {
        std::cout << "SELECTED PATH 3 -- normal conditions" << std::endl;
        normalConditionsPrediction(finalPairs);
    }

    return finalPairs;
}

void Leftover::normalConditionsPrediction(std::vector<Couple> &finalPairs)
{
    // FOR EVERY ORIGINAL DISH     -->     4 PREDICTED LEFTOVERS
    // check how many predictions are the same

    std::sort(pairAvgColors.begin(), pairAvgColors.end(), [](const Couple &a, const Couple &b)
              { return a.dist < b.dist; });

    std::vector<Couple> temporaryPair;
    bool noOtherPerfect = true;

    int keepCount = 0;
    for (int i = 0; i < pairMatches.size(); i++)
    {
        // initialize counter for equal results
        int counterEquals = 0;
        // Count number of same results from all measurement methods
        if (checkCouplesEqual(pairMatches[i], pairAvgColors[i]))
            counterEquals += 1;
        if (checkCouplesEqual(pairMatches[i], pairArea[i]))
            counterEquals += 1;
        if (checkCouplesEqual(pairMatches[i], pairCieAvgs[i]))
            counterEquals += 1;

        // if all four couple are the same
        if (counterEquals == 3)
        {
            keepCount++;
            if ((pairMatches.size() == 3 && keepCount == 3) || (pairMatches.size() == 2 && keepCount == 2))
            {
                finalPairs.push_back(pairAvgColors[0]);
                noOtherPerfect = false;
            }
            else
            {
                temporaryPair.push_back(pairMatches[i]);
            }
        }

        // if three couples are the same
        else if (counterEquals == 2)
            finalPairs.push_back(pairMatches[i]);

        // if two couples are the same and two different between each others
        else if (counterEquals == 1)
        {
            // The other two couples are equal but different from pairMatches
            // There is 50% chance for both pairMatches[i] and two of the other metrics.
            if (checkCouplesEqual(pairAvgColors[i], pairArea[i]) || checkCouplesEqual(pairAvgColors[i], pairCieAvgs[i]) || checkCouplesEqual(pairArea[i], pairCieAvgs[i]))
            {
                if (checkCouplesEqual(pairAvgColors[i], pairArea[i]) || checkCouplesEqual(pairAvgColors[i], pairCieAvgs[i]))
                    finalPairs.push_back(pairAvgColors[i]);
                else
                    finalPairs.push_back(pairCieAvgs[i]);
            }
            // The other two couples are equal but different between each others
            // it is still 50% chance that the predictions are good
            else
                finalPairs.push_back(pairMatches[i]);
        }
        // if all couples are different -- Count == 0
        else
            finalPairs.push_back(pairAvgColors[i]);
    }

    if (noOtherPerfect == true && temporaryPair.size() > 0)
    {
        std::cout << "No other perfect and Counter equals = 3\n";
        finalPairs.push_back(temporaryPair[0]);
    }
}

void Leftover::moreOriginalLessLeftovers(int type, std::vector<Couple> &finalPairs, std::vector<cv::Mat> &alreadyAssigned)
{
    if (type == 1)
    {

        std::sort(pairCieAvgs.begin(), pairCieAvgs.end(), [](const Couple &a, const Couple &b)
                  { return a.dist < b.dist; });
        for (int i = 0; i <= 1; i++)
        {
            finalPairs.push_back(pairCieAvgs[i]);
        }

        // Match with black picture
        for (int i = 1; i < originalDishes.size(); i++)
        {
            Couple emptyCouple;
            emptyCouple.leftover = cv::Mat::zeros(originalDishes[i].size(), originalDishes[i].type());
            std::string errMessage = "No matches found for this dish.";
            cv::putText(emptyCouple.leftover, errMessage, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            emptyCouple.original = originalDishes[i];
            emptyCouple.empty = true;
            finalPairs.push_back(emptyCouple);
        }
    }
    else if (type == 2)
    {
        std::sort(pairCieAvgs.begin(), pairCieAvgs.end(), [](const Couple &a, const Couple &b)
                  { return a.dist < b.dist; });
        for (int i = 0; i < 2; i++)
        {
            finalPairs.push_back(pairCieAvgs[i]);
        }

        Couple emptyCouple;
        emptyCouple.leftover = cv::Mat::zeros(originalDishes[2].size(), originalDishes[2].type());
        std::string errMessage = "No matches found for this dish.";
        cv::putText(emptyCouple.leftover, errMessage, cv::Point(originalDishes[2].cols / 2.5, originalDishes[2].rows / 2.5), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        emptyCouple.original = pairAvgColors[2].original;
        emptyCouple.empty = true;

        finalPairs.push_back(emptyCouple);
    }
}
// ------------------------------------------------------------------------------------------------------ //
/*
    Computes Original and Leftover with similar CIELAB colors.
    Distance can be computed with DeltaE method or with norm method as for the RGB average method.
    @returns: vector of Couple of Original,Leftover images with similar ranges in CIELAB space.
    Flag = 0 --> DeltaE
    Flag = 1 --> Norm
*/
std::vector<Couple> Leftover::coupleCIELABColors(const std::vector<cv::Mat> &originals, const std::vector<cv::Mat> &leftovers, bool flag)
{
    std::vector<Couple> result;
    std::vector<bool> used(leftovers.size(), false);
    if (flag == 0)
    {
        for (int i = 0; i < avgCIELABOriginals.size(); i++)
        {
            const cv::Scalar &avgOriginal = avgCIELABOriginals[i];
            double minDist = std::numeric_limits<double>::max();
            int closestIndex = 0;
            for (int j = 0; j < avgCIELABLefts.size(); j++)
            {
                if (!used[j])
                {
                    const cv::Scalar &avgLeft = avgCIELABLefts[j];
                    double dist = computeDeltaE(avgOriginal, avgLeft);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        closestIndex = j;
                    }
                }
            }
            used[closestIndex] = true;
            Couple couple;
            couple.original = originals[i];
            couple.leftover = leftovers[closestIndex];
            couple.dist = minDist;
            result.push_back(couple);
        }
    }
    else if (flag == 1)
    {
        for (int i = 0; i < avgCIELABOriginals.size(); i++)
        {
            const cv::Scalar &avgOriginal = avgCIELABOriginals[i];
            double minDist = std::numeric_limits<double>::max();
            int closestIndex = 0;
            for (int j = 0; j < avgCIELABLefts.size(); j++)
            {
                if (!used[j])
                {
                    const cv::Scalar &avgLeft = avgCIELABLefts[j];
                    double dist = cv::norm(avgOriginal - avgLeft, cv::NORM_L2);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        closestIndex = j;
                    }
                }
            }
            used[closestIndex] = true;
            Couple couple;
            couple.original = originals[i];
            couple.leftover = leftovers[closestIndex];
            couple.dist = minDist;
            result.push_back(couple);
        }
    }
    return result;
}

/*
    Computes Original and Leftover with most matches
    @returns: Couple of Original,Leftover image with the least difference in area of circle
*/
Couple Leftover::coupleMaxMatches(const std::vector<int> &matches,
                                  std::vector<cv::Mat> &leftovers, const cv::Mat &original)
{
    if (matches.size() == 3)
    {
        if (matches[0] >= matches[1] && matches[0] >= matches[2])
        {
            Couple couple;
            couple.leftover = leftovers[0];
            couple.original = original;
            couple.matches = matches[0];

            return couple;
        }
        else if (matches[1] >= matches[0] && matches[1] >= matches[2])
        {
            Couple couple;
            couple.leftover = leftovers[1];
            couple.original = original;
            couple.matches = matches[1];

            return couple;
        }
        else if (matches[2] >= matches[0] && matches[2] >= matches[1])
        {
            Couple couple;
            couple.leftover = leftovers[2];
            couple.original = original;
            couple.matches = matches[2];

            return couple;
        }
    }
    else if (matches.size() == 2)
    {
        if (matches[0] > matches[1])
        {
            Couple couple;
            couple.leftover = leftovers[0];
            couple.original = original;
            couple.matches = matches[0];

            return couple;
        }
        else if (matches[1] > matches[0])
        {
            Couple couple;
            couple.leftover = leftovers[1];
            couple.original = original;
            couple.matches = matches[1];

            return couple;
        }
        else
        {
            int rand = std::rand() % 2;
            Couple couple;
            couple.leftover = leftovers[rand];
            couple.original = original;
            couple.matches = matches[rand];

            return couple;
        }
    }
    else if (matches.size() == 1)
    {
        Couple couple;
        // couple.leftover.create(original.size(), original.type());
        couple.leftover = leftovers[0];
        couple.original = original;
        couple.matches = matches[0];

        return couple;
    }
    else
    {
        std::cerr << "Number of matches --> " << matches.size() << std::endl;
    }
}

/*
    Computes Original and Leftover with least difference in area of circle
    @returns: vector of Couple of Original,Leftover image with the least difference in area of circle
    Not an accurate measurement but if it is used together with other measurements it can be useful.
*/
std::vector<Couple> Leftover::coupleClosestElements(const std::vector<cv::Mat> &originals,
                                                    const std::vector<cv::Mat> &leftovers)
{
    std::vector<Couple> result;
    std::vector<bool> used(leftovers.size(), false);
    for (int i = 0; i < circleAreasOriginal.size(); i++)
    {
        double original = circleAreasOriginal[i];
        double minDistance = std::numeric_limits<double>::max();
        int closestIndex = 0;
        for (int j = 0; j < circleAreasLeftover.size(); j++)
        {
            if (!used[j])
            {
                double leftover = circleAreasLeftover[j];
                double distance = std::abs(original - leftover);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestIndex = j;
                }
            }
        }
        used[closestIndex] = true;
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
    std::vector<bool> used(leftovers.size(), false);
    for (int i = 0; i < avgOriginals.size(); i++)
    {
        const cv::Scalar &avgOriginal = avgOriginals[i];
        double minDist = std::numeric_limits<double>::max();
        int closestIndex = 0;
        for (int j = 0; j < avgLefts.size(); j++)
        {
            if (!used[j])
            {
                const cv::Scalar &avgLeft = avgLefts[j];
                double dist = cv::norm(avgOriginal - avgLeft);
                if (dist < minDist)
                {
                    minDist = dist;
                    closestIndex = j;
                }
            }
        }
        used[closestIndex] = true;
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
    Assign a bounding box to every food in the dish
    This is done by reflection of the bounding boxes present in the original dish and
    passed as parameter with boxes.
    The result is then stored in results vector.
    @param: boxes
    @result: std::vector<Couple> segmentedPairsWithBBs
*/
std::vector<SegmentCouple> Leftover::createFinalPairs(const Dish &dish, const std::vector<Couple> &finalPairs)
{
    std::vector<SegmentCouple> finalVec;
    int i = 0;
    for (Couple fp : finalPairs)
    {
        ++i;
        SegmentCouple c;
        if (checkImageEqual(fp.original, dish.getDish()))
            c = createCouple(fp, dish, finalVec);
    }

    return finalVec;
}

SegmentCouple Leftover::createCouple(Couple c, Dish orig, std::vector<SegmentCouple> &finalVec)
{

    SegmentCouple result;
    ImageProcessor ip;

    cv::Mat originalDish = c.original;
    cv::Mat leftoverDish = c.leftover;

    std::cout << "Num Foods " << orig.getFoods().size() << std::endl;

    int i = 0;
    for (FoodData f : orig.getFoods())
    {
        ++i;
        cv::Mat segmentOriginal;

        originalDish.copyTo(segmentOriginal, f.segmentArea);
        cv::Scalar avgColorOriginal = computeAvgColor(segmentOriginal);

        int k = orig.getFoods().size() + 1;

        cv::Mat shifted;
        bilateralFilter(leftoverDish, shifted, 1, 0.5, 0.5);
        cv::pyrMeanShiftFiltering(shifted, shifted, 40, 200);
        cv::Mat segmentLeftover = ip.kmeansSegmentation(k, shifted);

        if (k == 2)
        {
            result.id = f.id;
            result.segmentOriginal = f.segmentArea;
            result.segmentLeftover = getYellowArea(segmentLeftover);

            std::cout << "K == 2\n";
            std::cout << "Result " << i << " Segment original size: " << result.segmentOriginal.size() << std::endl;
            std::cout << "Result " << i << " Segment Leftover size: " << result.segmentLeftover.size() << std::endl;
            std::cout << "Result " << i << " Food id: " << result.id << std::endl;

            finalVec.push_back(result);
        }
        if (k == 3)
        {
            cv::Mat maskYellow = getYellowArea(segmentLeftover);
            cv::Mat maskBlue = getBlueArea(segmentLeftover);

            cv::Mat segment1, segment2;
            leftoverDish.copyTo(segment1, maskYellow);
            leftoverDish.copyTo(segment2, maskBlue);

            cv::Scalar avgSegment1, avgSegment2;
            avgSegment1 = computeAvgColor(segment1);
            avgSegment2 = computeAvgColor(segment2);

            double dist1 = cv::norm(avgColorOriginal - avgSegment1);
            double dist2 = cv::norm(avgColorOriginal - avgSegment2);

            if (dist1 < dist2)
            {
                result.segmentLeftover = maskYellow;
                result.segmentOriginal = f.segmentArea;
                result.id = f.id;
            }
            else
            {
                result.segmentLeftover = maskBlue;
                result.segmentOriginal = f.segmentArea;
                result.id = f.id;
            }

            std::cout << "K == 3\n";
            std::cout << "Result " << i << " Segment original size: " << result.segmentOriginal.size() << std::endl;
            std::cout << "Result " << i << " Segment Leftover size: " << result.segmentLeftover.size() << std::endl;
            std::cout << "Result " << i << " Food id: " << result.id << std::endl;

            finalVec.push_back(result);
        }
    }

    return result;
}

void Leftover::breadFinder(cv::Mat &leftover)
{
    cv::Mat final = leftover.clone(); // Final image - Result
    std::vector<FoodData> bboxes;     // Bounding Boxes
    int area = boundBreadLeftover(leftover, leftoverDishes, final, bboxes);
    std::cout << "final size: " << final.size() << std::endl;
    drawBoundingBoxes(final, bboxes);
    showImg("pane", final);
}

// ------------------------------------------------------------------------------------------------------ //
/*
    Checks if the Original is the same as the Original predicted
    and if the Leftover is the same as the Leftover predicted
*/
bool checkCouplesEqual(const Couple &a, const Couple &b)
{
    cv::Mat aOriginalGray, bOriginalGray, aLeftoverGray, bLeftoverGray;
    // For same original image in both couples
    cv::cvtColor(a.original, aOriginalGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b.original, bOriginalGray, cv::COLOR_BGR2GRAY);
    // For same leftover image in both couples
    cv::cvtColor(a.leftover, aLeftoverGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(b.leftover, bLeftoverGray, cv::COLOR_BGR2GRAY);
    // Actual check
    return cv::countNonZero(aOriginalGray != bOriginalGray) == 0 && cv::countNonZero(aLeftoverGray != bLeftoverGray) == 0;
}

bool checkImageEqual(const cv::Mat &a, const cv::Mat &b)
{
    if (a.empty() || b.empty())
    {
        std::cerr << "CheckImageEqual, error with passed images.\n";
        return false;
    }
    else
    {
        cv::Mat aOriginalGray, bOriginalGray;
        std::cout << "size a: " << a.size << "\n";
        std::cout << "size b: " << b.size << "\n";

        cv::cvtColor(a, aOriginalGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(b, bOriginalGray, cv::COLOR_BGR2GRAY);

        std::cout << "size a: " << aOriginalGray.size << "\n";
        std::cout << "size b: " << bOriginalGray.size << "\n";

        return cv::countNonZero(aOriginalGray != bOriginalGray) == 0;
    }
}

void printVector(const std::vector<Couple> &pairs, const std::string &title)
{
    std::cout << "\nentro nel print vector\n";
    if (pairs.size() == 0)
    {
        std::cerr << "Pairs vuoto. ";
    }

    int i = 0;
    for (const auto &pair : pairs)
    {
        std::cout << "i: " << i << "; ";
        i++;
        concatShowImg(title, pair.original, pair.leftover);
    }
}

void printVectorUpdate(const std::vector<SegmentCouple> &pairs, const std::string &title)
{
    std::cout << "\nentro nel print vector updated\n";
    if (pairs.size() == 0)
    {
        std::cerr << "Pairs vuoto. ";
    }

    int i = 0;
    for (const auto &pair : pairs)
    {
        std::cout << "i: " << i << "; ";
        i++;
        std::string ftitle = title + std::to_string(pair.id);
        if (!pair.segmentOriginal.empty() && !pair.segmentLeftover.empty())
            concatShowImg(ftitle, pair.segmentOriginal, pair.segmentLeftover);
        else
        {
            std::cerr << "One of the two pairs is empty.";
        }
    }
}

double computeDeltaE(const cv::Scalar &c1, const cv::Scalar &c2)
{
    double deltaL = c1[0] - c2[0];
    double deltaA = c1[1] - c2[1];
    double deltaB = c1[2] - c2[2];
    return std::sqrt(deltaL * deltaL + deltaA * deltaA + deltaB * deltaB);
}
