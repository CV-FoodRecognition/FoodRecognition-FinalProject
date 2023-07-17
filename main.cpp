#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImagePreprocessor.hpp"
#include "headers/ImageProcessor.h"
#include "headers/Leftover.h"
#include "headers/metrics.h"

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
    addFood(2, "rice", "pilaw rice", 5, "../images/Train/", templates);
    addFood(3, "pork", "grilled pork cutlet", 6, "../images/Train/", templates);
    addFood(2, "fishcutlet", "fish cutlet", 7, "../images/Train/", templates);
    addFood(2, "rabbit", "rabbit", 8, "../images/Train/", templates);
    addFood(2, "seafoodsalad", "seafood salad", 9, "../images/Train/", templates); // problems
    addFood(2, "beans", "beans", 10, "../images/Train/", templates);
    addFood(0, "bread", "bread", 13, "../images/Train/", templates);
    addFood(2, "potatoes", "basil potatoes", 11, "../images/Train/", templates);
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

    /*
        1st method:
        Detect and Recognize Objects
    */
    // showImg("0", templates[0]);
    // showImg("1", templates[1]);

    cv::Mat final = in1.clone();
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

    // detectAndRecognize(dishes, templates, dishesMatches, in1, final, result, acceptedCircles);
    // showImg("FINALE", final);

    cout << "XX" << endl;

    /* // READING RESULTS
   for (int d = 0; d < dishes.size(); d++)
   {
       Mat segmentedImg = imread("../images/Results/kmeansResult" + to_string(d) + ".jpg", CV_32F);
       segmentedImages.push_back(segmentedImg);
   } */

    Leftover leftover;
    leftover.matchLeftovers(removedDishes, leftovers, leftoverImg, radia1, radia2);

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

void detectAndRecognize(std::vector<cv::Mat> &dishes, std::vector<foodTemplate> &templates,
                        std::vector<int> &dishesMatches, cv::Mat &in1, cv::Mat &final, Result &result,
                        std::vector<cv::Vec3f> &accepted_circles)
{
    std::vector<int> forbidden;
    bool alreadyFoundFirstDish = false;
    std::vector<BoundingBox> boundingBoxes;

    for (int f = 0; f < templates.size(); f++)
    { // for every food

        for (int d = 0; d < dishes.size(); d++)
            dishesMatches[d] = 0;

        cout << "inizio a cercare match" << endl;

        for (int t = 0; t < templates[f].foodTemplates.size(); t++)
        { // for every food template
            cout << "t: " << t << endl;

            for (int d = 0; d < dishes.size(); d++)
            { // look for matches in every dish
                cout << "entro qui" << endl;
                cout << "d: " << d << endl;
                result = useDescriptor(dishes[d], templates[f].foodTemplates[t], DescriptorType::SIFT);
                dishesMatches[d] = dishesMatches[d] + bruteForceKNN(dishes[d], templates[f].foodTemplates[t], result);
                std::cout << "matches dishes " << d << " : " << dishesMatches[d] << std::endl;
            }
        }
        cout << "finito di cercare i match" << endl;

        int max_key = 0;
        Scalar avgColor;
        if (templates[f].id != 13)
        {
            max_key = computeBestDish(templates[f], dishes, dishesMatches);
            cout << "max key: " << max_key << "avg color prima" << endl;

            if (max_key > -1)
            {
                avgColor = computeAvgColorHSV(dishes[max_key]);
            }
        }

        cout << "trovato il piatto con più match" << endl;

        if (max_key > -1)
        {
            if (templates[f].id == 1 && avgColor[0] > 30 && avgColor[0] < 50 && !alreadyFoundFirstDish)
            {
                // pesto
                boundPasta(dishes[max_key], final, templates[f].label, forbidden, max_key, boundingBoxes);
                forbidden.push_back(max_key);
                alreadyFoundFirstDish = true;
            }
            else if ((templates[f].id == 2 || templates[f].id == 3) && avgColor[0] > 0 && avgColor[0] < 11 && !alreadyFoundFirstDish)
            { // tomato
                std::vector<std::string> labels;
                labels.push_back("pasta with tomato sauce");
                labels.push_back("pasta with meat sauce");
                boundPasta(dishes[max_key], final, labels, forbidden, max_key, boundingBoxes);
                forbidden.push_back(max_key);
                alreadyFoundFirstDish = true;
            }
            else if (templates[f].id == 4 && avgColor[0] > 10 && avgColor[0] < 20 && !alreadyFoundFirstDish)
            { // clums and mussels
                boundPasta(dishes[max_key], final, templates[f].label, forbidden, max_key, boundingBoxes);
                forbidden.push_back(max_key);
                alreadyFoundFirstDish = true;
            }
            else if (templates[f].id == 12)
            { // salad
                boundSalad(in1, accepted_circles, final, boundingBoxes);
            }
            else if (templates[f].id == 13)
            { // bread
                boundBread(in1, dishes, final, boundingBoxes);
            }
            else if (templates[f].id > 4)
            {
                bool allowed = true;
                for (int i = 0; i < forbidden.size(); i++)
                {
                    if (forbidden[i] == max_key)
                    {
                        allowed = false;
                        break;
                    }
                }
                if (allowed)
                {
                    bruteForceKNN(in1, templates[f], dishes[max_key], final, boundingBoxes);
                }
            }
        }
    }
    drawBoundingBoxes(final, boundingBoxes);
    return;
}
