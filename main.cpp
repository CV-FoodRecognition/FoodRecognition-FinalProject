#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImageProcessor.h"
#include "headers/Leftover.h"
#include "headers/metrics.h"
#include "headers/detect_recognition.h"

/*
Written by @nicolacalzone and @rickyvendra
*/

using namespace cv;
using namespace std;

const std::string window_name = "K Means Trackbar";
const int max_k = 5;
int low_k = 1;

void computeProbability(BoxLabel &box);
void computeSegmentArea(SegmentAreas &sa);
void detectAndRecognize(std::vector<cv::Mat> &dishes, std::vector<foodTemplate> &templates,
                        std::vector<int> &dishesMatches, cv::Mat &in1, cv::Mat &final, Result &result,
                        std::vector<cv::Vec3f> &accepted_circles);
static void onTrackbar(int, void *user);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path> <leftover_path>" << std::endl;
        return -1;
    }

    std::string nameFile1 = "../images/" + std::string(argv[1]);
    std::string nameFile2 = "../images/Leftovers/" + std::string(argv[2]);

    std::vector<cv::Mat> segmentedImages;
    Result result;
    std::vector<foodTemplate> templates;

    // Read Input
    cv::Mat in1 = cv::imread(nameFile1, CV_32F);
    if (!in1.data)
    {
        std::cerr << "ERROR on input image" << std::endl;
        return -1;
    }
    // Read Leftover
    cv::Mat leftoverImg = cv::imread(nameFile2, cv::IMREAD_COLOR);
    if (!leftoverImg.data)
    {
        std::cerr << "ERROR on leftover input image" << std::endl;
        return -1;
    }

    addFood(0, "", "pasta with pesto", 1, "../images/Train/", templates);
    addFood(0, "", "pasta with tomato sauce", 2, "../images/Train/", templates);
    addFood(0, "", "pasta with meat sauce", 3, "../images/Train/", templates);
    addFood(0, "", "pasta with clams and mussels", 4, "../images/Train/", templates); // problems
    addFood(4, "rice", "pilaw rice", 5, "../images/Train/", templates);
    addFood(3, "pork", "grilled pork cutlet", 6, "../images/Train/", templates);
    addFood(2, "fishcutlet", "fish cutlet", 7, "../images/Train/", templates);
    addFood(2, "rabbit", "rabbit", 8, "../images/Train/", templates);
    addFood(2, "seafoodsalad", "seafood salad", 9, "../images/Train/", templates); // problems
    addFood(2, "beans", "beans", 10, "../images/Train/", templates);
    addFood(0, "bread", "bread", 13, "../images/Train/", templates);
    addFood(0, "potatoes", "basil potatoes", 11, "../images/Train/", templates);
    addFood(0, "salad", "salad", 12, "../images/Train/", templates); // problems

    // Hough Transform
    ImageProcessor imgProc;
    imgProc.doHough(in1);
    std::vector<int> &dishesMatches = imgProc.getDishesMatches();
    std::vector<cv::Mat> &dishes = imgProc.getDishes();
    std::vector<int> &radia1 = imgProc.getRadius();
    std::vector<cv::Vec3f> &acceptedCircles = imgProc.getAcceptedCircles();

    // Hough Transform 2
    ImageProcessor imgProcLeftovers;
    imgProcLeftovers.doHough(leftoverImg);
    std::vector<cv::Mat> &leftovers = imgProcLeftovers.getDishes();
    std::vector<int> &radia2 = imgProcLeftovers.getRadius();

    cv::Mat final = in1.clone();
    std::vector<FoodData> foodData;
    detectAndCompute(in1, dishes, dishesMatches, acceptedCircles, foodData, templates, final);
    showImg("Detect and Recognize", final);

    std::vector<cv::Mat> removedDishes;
    for (int d = 0; d < dishes.size(); d++)
    {
        // FILTERS
        cv::Mat src = dishes[d];
        cv::Mat rmvDish = dishes[d];
        // cv::Mat shifted;
        // bilateralFilter(src, shifted, 1, 0.5, 0.5);
        // cv::pyrMeanShiftFiltering(shifted, shifted, 40, 200);
        // showImg("PyrMean", shifted);
        // removeDish(shifted);

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

    cout << "XX" << endl;

    /* // READING RESULTS
   for (int d = 0; d < dishes.size(); d++)
   {
       Mat segmentedImg = imread("../images/Results/kmeansResult" + to_string(d) + ".jpg", CV_32F);
       segmentedImages.push_back(segmentedImg);
   } */

    Leftover leftover;

    /*std::vector<cv::Mat> inputDishes;
    for (const auto &dish : dishesWithBB)
        inputDishes.push_back(dish.first);*/

    leftover.matchLeftovers(removedDishes, leftovers, leftoverImg, radia1, radia2, foodData);

    cout << "\n---------------\nfine leftovers" << endl;

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

    return 0;
}
