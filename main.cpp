#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImagePreprocessor.hpp"

using namespace cv;
using namespace std;

const std::string window_name = "K Means Trackbar";
const int max_k = 5;
int low_k = 1;

void computeProbability(BoxLabel &box);
void computeSegmentArea(SegmentAreas &sa);
void detectAndRecognize(std::vector<cv::Mat> &dishes, std::vector<cv::Mat> &templates,
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
    cv::Mat in1 = cv::imread(nameFile, CV_32F);
    Result result;
    std::vector<cv::Mat> templates;
    std::vector<std::string> labels;

    // READ TEMPLATES
    Mat templ_fagioli = cv::imread("../images/Train/beans.jpg", cv::IMREAD_GRAYSCALE);
    Mat templ_conchiglie = cv::imread("../images/Train/conchiglie.jpg", cv::IMREAD_GRAYSCALE);
    templates.push_back(templ_fagioli);
    labels.push_back("fagioli");
    templates.push_back(templ_conchiglie);
    labels.push_back("pasta");

    if (!in1.data)
    {
        std::cerr << "ERROR on input image" << std::endl;
        return -1;
    }

    // Hough Transform
    std::vector<cv::Mat> dishes;
    std::vector<int> dishesMatches;
    doHough(dishes, dishesMatches, in1);

    for (int i = 0; i < dishesMatches.size(); i++)
        cout << dishesMatches[i];

    /* 1st method:
        Detect and Recognize Objects
    */
    // showImg("0", templates[0]);
    // showImg("1", templates[1]);

    cv::Mat final = in1.clone();
    // detectAndRecognize(dishes, templates, dishesMatches, in1, final, result);

    /*  2nd method:
        Detect and Recognize Objects
    */
    std::vector<cv::Rect> mser_boxes;
    for (int d = 0; d < dishes.size(); d++)
    {
        // FILTERS
        cv::Mat src = dishes[d];
        cv::Mat shifted, bilateral;
        bilateralFilter(src, shifted, 1, 0.5, 0.5);
        cv::pyrMeanShiftFiltering(src, shifted, 40, 200);
        // showImg("PyrMean", shifted);
        removeDish(shifted);
        // sharpenImg(shifted, SharpnessType::LAPLACIAN);

        // CALLBACK
        /* namedWindow(window_name);
        PassedStruct *ps = new PassedStruct;
        ps->p1 = shifted;
        ps->p2 = to_string(d);
        createTrackbar("K trackbar", window_name, NULL, max_k, onTrackbar, ps);
        onTrackbar(2, ps);
        waitKey(0);
        delete ps; */
        showImg("Choose a K for KMeans", shifted);
        int k;
        cout << "Choose a K KMeans: ";
        cin >> k;
        cv::Mat r = kmeansSegmentation(k, shifted);
        showImg(to_string(k), r);
        imwrite("../images/Results/kmeansResult" + to_string(d) + ".jpg", r);
    }

    // READING RESULTS
    for (int d = 0; d < dishes.size(); d++)
    {
        Mat segmentedImg = imread("../images/Results/kmeansResult" + to_string(d) + ".jpg", CV_32F);
        segmentedImages.push_back(segmentedImg);
    }

    // SEGMENTATION
    for (int i = 0; i < segmentedImages.size(); i++)
    {
        if (segmentedImages[i].data)
        {
            SegmentAreas sa;
            sa.p1 = segmentedImages[i];
            showImg("aa", sa.p1);
            computeSegmentArea(sa);
            cout << "Area Blu: " << sa.areaBlue << "\nArea gialla: " << sa.areaYellow
                 << "Area verde: " << sa.areaGreen << "\nArea rossa: " << sa.areaRed
                 << "Area nera: " << sa.areaBlack << endl;
        }
    }

    /*
        Compute probability for objects
    */
    // computeProbability();

    return 0;
}

void detectAndRecognize(std::vector<cv::Mat> &dishes, std::vector<cv::Mat> &templates,
                        std::vector<int> &dishesMatches, cv::Mat &in1, cv::Mat &final, Result &result)
{
    for (int t = 0; t < templates.size(); t++)
    {
        cout << "inizio a cercare match" << endl;
        for (int i = 0; i < dishes.size(); i++)
        {
            std::cout << "result\n";

            result = useDescriptor(dishes[i], templates[i], DescriptorType::SIFT);
            std::cout << "result\n";
            cv::Mat some = cv::Mat(in1.rows, in1.cols, CV_8UC3);
            std::cout << "some\n";

            dishesMatches[i] = bruteForceKNN(dishes[i], templates[i], result, some);
            std::cout << "matches dishes " << i << " : " << dishesMatches[i] << std::endl;
        }
        cout << "finito di cercare i match" << endl;
        int max_key = 0;
        for (int i = 0; i < dishesMatches.size(); i++)
            if (dishesMatches[i] > dishesMatches[max_key])
                max_key = i;
        cout << "trovato il piatto con più match" << endl;
        if (dishesMatches[max_key] > 0)
        {
            cout << "match finale (quello più giusto)" << endl;
            result = useDescriptor(dishes[max_key], templates[t], DescriptorType::SIFT);
            bruteForceKNN(in1, templates[t], result, final);
        }
    }
}

void computeSegmentArea(SegmentAreas &sa)
{
    Mat maskYellow, maskBlue, maskGreen, maskRed, maskBlack;
    inRange(sa.p1, Scalar(0, 255, 255), Scalar(0, 255, 255), maskYellow);
    inRange(sa.p1, Scalar(255, 0, 0), Scalar(255, 0, 0), maskBlue);
    inRange(sa.p1, Scalar(0, 255, 0), Scalar(0, 255, 0), maskGreen);
    inRange(sa.p1, Scalar(0, 0, 255), Scalar(0, 0, 255), maskRed);
    inRange(sa.p1, Scalar(0, 0, 0), Scalar(10, 10, 10), maskBlack);

    sa.areaYellow = countNonZero(maskYellow);
    sa.areaBlue = countNonZero(maskBlue);
    sa.areaGreen = countNonZero(maskGreen);
    sa.areaRed = countNonZero(maskRed);
    sa.areaBlack = countNonZero(maskBlack);
}

void computeProbability(BoxLabel &box)
{
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

static void onTrackbar(int, void *user)
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
}