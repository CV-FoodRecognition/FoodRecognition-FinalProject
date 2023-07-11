#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImagePreprocessor.hpp"
#include "headers/ImageProcessor.h"
#include "headers/leftover.h"

using namespace cv;
using namespace std;

const std::string window_name = "K Means Trackbar";
const int max_k = 5;
int low_k = 1;

void computeProbability(BoxLabel &box);
void computeSegmentArea(SegmentAreas &sa);
void detectAndRecognize(std::vector<cv::Mat> &dishes, std::vector<foodTemplate> &templates,
                        std::vector<int> &dishesMatches, cv::Mat &in1, cv::Mat &final, Result &result);
static void onTrackbar(int, void *user);

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string nameFile = "../images/" + std::string(argv[1]);
    std::vector<cv::Mat> segmentedImages;
    Result result;
    std::vector<foodTemplate> templates;

    cv::Mat in1 = cv::imread(nameFile, CV_32F);
    if (!in1.data)
    {
        std::cerr << "ERROR on input image" << std::endl;
        return -1;
    }

    addFood(2, "beans", "beans", 10, "../images/Train/", templates);

    // Hough Transform
    ImageProcessor imgProc;
    imgProc.doHough(in1);
    std::vector<int> &dishesMatches = imgProc.getDishesMatches();
    std::vector<cv::Mat> &dishes = imgProc.getDishes();
    std::vector<int> &radia1 = imgProc.getRadius();

    // Read Leftover
    cv::Mat leftoverImg = cv::imread("../images/Leftovers/leftover1_3.jpg", cv::IMREAD_COLOR);

    // Hough Transform 2
    ImageProcessor imgProcLeftovers;
    imgProcLeftovers.doHough(leftoverImg);
    std::vector<cv::Mat> &leftovers = imgProcLeftovers.getDishes();
    std::vector<int> &radia2 = imgProcLeftovers.getRadius();

    /* 1st method:
        Detect and Recognize Objects
    */
    // showImg("0", templates[0]);
    // showImg("1", templates[1]);

    cv::Mat final = in1.clone();

    /*  2nd method:
        Detect and Recognize Objects
    */

    cv::Mat resMSER;

    std::vector<cv::Mat> removedDishes;
    for (int d = 0; d < dishes.size(); d++)
    {
        // FILTERS
        cv::Mat src = dishes[d];
        cv::Mat rmvDish = dishes[d];
        cv::Mat shifted;
        bilateralFilter(src, shifted, 1, 0.5, 0.5);
        cv::pyrMeanShiftFiltering(shifted, shifted, 40, 200);
        // showImg("PyrMean", shifted);
        removeDish(shifted);

        removeDish(rmvDish);
        sharpenImg(rmvDish, SharpnessType::LAPLACIAN);

        removedDishes.push_back(rmvDish);
        // showImg("Image", rmvDish);

        // imgProc.doMSER(shifted, resMSER);
        // showImg("MSER", resMSER);

        // CALLBACK
        /*namedWindow(window_name);
        PassedStruct *ps = new PassedStruct;
        ps->p1 = shifted;
        ps->p2 = to_string(d);
        createTrackbar("K trackbar", window_name, NULL, max_k, onTrackbar, ps);
        onTrackbar(2, ps);
        waitKey(0);
        delete ps;

        showImg("Choose a K for KMeans", shifted);
        int k;
        cout << "Choose a K KMeans (max 5): ";
        cin >> k;
        k = min(5, k);

        cv::Mat r = imgProc.kmeansSegmentation(k, shifted);
        showImg(to_string(k), r);
        imwrite("../images/Results/kmeansResult" + to_string(d) + ".jpg", r); */
    }

    detectAndRecognize(dishes, templates, dishesMatches, in1, final, result);
    showImg("FINALE", final);

    cout << "XX" << endl;

    /* // READING RESULTS
   for (int d = 0; d < dishes.size(); d++)
   {
       Mat segmentedImg = imread("../images/Results/kmeansResult" + to_string(d) + ".jpg", CV_32F);
       segmentedImages.push_back(segmentedImg);
   } */

    computeLeftovers(removedDishes, leftovers, radia1, radia2);

    cout << "XXX" << endl;

    // SEGMENTATION
    /* for (int i = 0; i < segmentedImages.size(); i++)
    {
        if (segmentedImages[i].data)
        {
            SegmentAreas sa;
            sa.p1 = segmentedImages[i];
            showImg("aa", sa.p1);
            computeSegmentArea(sa);
            cout << "Area Blu: " << sa.areaBlue << "\nArea gialla: " << sa.areaYellow
                 << "\nArea verde: " << sa.areaGreen << "\nArea rossa: " << sa.areaRed
                 << "\nArea nera: " << sa.areaBlack << endl;
        }
    } */

    /*
        Compute probability for objects
    */
    // computeProbability();

    return 0;
}

void detectAndRecognize(std::vector<cv::Mat> &dishes, std::vector<foodTemplate> &templates,
                        std::vector<int> &dishesMatches, cv::Mat &in1, cv::Mat &final, Result &result)
{
    for (int f = 0; f < templates.size(); f++)
    { // for every food
        for (int d = 0; d < dishes.size(); d++)
            dishesMatches[d] = 0;

        cout << "inizio a cercare match" << endl;

        for (int t = 0; t < templates[f].foodTemplates.size(); t++)
        { // for every food template
            for (int d = 0; d < dishes.size(); d++)
            { // look for matches in every dish
                result = useDescriptor(dishes[d], templates[f].foodTemplates[t], DescriptorType::SIFT);
                dishesMatches[d] = dishesMatches[d] + bruteForceKNN(dishes[d], templates[f].foodTemplates[t], result);
                std::cout << "matches dishes " << d << " : " << dishesMatches[d] << std::endl;
            }
            cout << "finito di cercare i match" << endl;
        }
        int max_key = 0;
        for (int d = 0; d < dishesMatches.size(); d++)
        {
            if (dishesMatches[d] > dishesMatches[max_key])
            {
                max_key = d;
            }
        }
        cout << "trovato il piatto con più match" << endl;
        if (dishesMatches[max_key] > 0)
        {
            cout << "match finale (quello più giusto)" << endl;
            bruteForceKNN(in1, templates[f], dishes[max_key], final);
        }
    }
}

void computeProbability(BoxLabel &box)
{
    // if box area is tot
    // && avgColor is tot
    // && box area on segmentation yellow

    // if box area is tot
    // && avgColor is tot
    // && box area on segmentation blue

    Mat out;
    Scalar upperBound(200, 170, 180); // Lighter Part of Meat
    Scalar lowerBound(95, 80, 60);    // Darker Part of Meat

    bool inRange = true;
    for (int i = 0; i < 3; i++)
    {
        if (box.averageBoxColor[i] < lowerBound[i] || box.averageBoxColor[i] > upperBound[i])
        {
            inRange = false;
            break;
        }
    }

    cout << inRange;

    if (box.areaBox > 2000 && inRange)
        box.label = FoodType::Meat;
    else
        box.label = FoodType::Beans;
}

/* static void onTrackbar(int, void *user)
{
    PassedStruct &ps = *(PassedStruct *)user;
    cv::Mat src = ps.p1;
    std::string count = ps.p2;

    cv::Mat out;
    cv::Mat srcCopy = src.clone();
    int k = cv::getTrackbarPos("K trackbar", window_name);
    if (k > 0)
    {
        out = kmeansSegmentation(k, srcCopy);
        imshow(window_name, out);
        imwrite("../images/Results/kmeansResult" + count + ".jpg", out);
    }
} */