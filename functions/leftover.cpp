#include "../headers/leftover.h"
#include "../headers/utils.h"
#include "../headers/descriptor_methods.h"
#include "../headers/matcher_methods.h"

void computeLeftovers(std::vector<cv::Mat> &removedDishes, const std::vector<cv::Mat> &leftovers)
{

    // leggi 1 piatto
    cv::Mat piatto1 = removedDishes[0];
    // leggi 2 piatto
    cv::Mat piatto2 = removedDishes[1];
    // leggi 3 piatto
    cv::Mat piatto3;
    if (removedDishes.size() == 3)
        piatto3 = removedDishes[2];

    std::cout << "A" << endl;

    std::vector<cv::Mat> removedLeftovers;
    for (int d = 0; d < leftovers.size(); d++)
    {
        cv::Mat src = leftovers[d];
        cv::Mat shifted, bilateral;
        bilateralFilter(src, shifted, 1, 0.5, 0.5);
        cv::pyrMeanShiftFiltering(shifted, shifted, 40, 200);
        // showImg("PyrMean", shifted);
        removeDish(shifted);
        sharpenImg(shifted, SharpnessType::LAPLACIAN);

        removedLeftovers.push_back(shifted);
    }

    std::cout << "AA" << endl;

    // Calcola avg color 1
    Scalar avg1 = computeAvgColor(piatto1);

    // Calcola avg color 2
    Scalar avg2 = computeAvgColor(piatto2);

    // Calcola avg color 3
    Scalar avg3;
    if (removedDishes.size() == 3)
        avg3 = computeAvgColor(piatto3);

    double distMaxLevel1, distMaxLevel2, distMaxLevel3;

    Result res1, res2, res3;
    Couple couple;

    // Calcola sift 1 - 2
    for (int i = 0; i < removedDishes.size(); i++)
    {
        res1 = useDescriptor(removedDishes[i], removedLeftovers[0], DescriptorType::SIFT);
        res2 = useDescriptor(removedDishes[i], removedLeftovers[1], DescriptorType::SIFT);

        int matches1 = bruteForceKNN(removedDishes[i], removedLeftovers[0], res1);
        int matches2 = bruteForceKNN(removedDishes[i], removedLeftovers[1], res2);

        std::cout << "AAA" << endl;

        if (removedDishes.size() == 3)
        {
            res3 = useDescriptor(removedDishes[i], removedLeftovers[2], DescriptorType::SIFT);
            int matches3 = bruteForceKNN(removedDishes[i], removedLeftovers[2], res3);
            couple = computeMax(matches1, matches2, matches3, leftovers, removedDishes[i]);
        }
        else
            couple = computeMax(matches1, matches2, leftovers, removedDishes[i]);

        std::cout << "AAAA" << endl;

        showImg("original", couple.original);
        showImg("leftover", couple.leftover);

        std::cout << "AAAA" << endl;
    }

    // calcola area cerchio 1

    // calcola area cerchio 2

    // se avgcl1 simile ad avgcl2 && area simile (10 pixel max) && num match è ok
    firstLevel();

    // se avgcl1 simile (ma meno) && area simile MA num match basso
    // distMax avgcl1 avgcl2 maggiore di prima
    secondLevel();

    // se avgcl1 diverso || area diversa && num match basso
    thirdLevel();
}

Couple computeMax(int matches1, int matches2, std::vector<cv::Mat> leftovers, const cv::Mat &original)
{
    if (matches1 >= matches2)
    {
        Couple couple;
        couple.leftover = leftovers[0];
        couple.original = original;
        return couple;
    }
    else
    {
        Couple couple;
        couple.leftover = leftovers[1];
        couple.original = original;
        return couple;
    }
}

Couple computeMax(int matches1, int matches2, int matches3, std::vector<cv::Mat> leftovers, const cv::Mat &original)
{
    if (matches1 >= matches2 && matches1 >= matches3)
    {
        Couple couple;
        couple.leftover = leftovers[0];
        couple.original = original;
        return couple;
    }
    else if (matches2 >= matches1 && matches2 >= matches3)
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

void firstLevel()
{
}

void secondLevel() {}

void thirdLevel() {}
