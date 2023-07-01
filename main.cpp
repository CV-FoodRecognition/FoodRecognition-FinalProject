#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImagePreprocessor.hpp"

using cv::Vec3b, cv::Rect, cv::Scalar, cv::Point, cv::Mat;
using std::string, std::vector;

using namespace cv;
using namespace std;

std::string window_name = "K Means Trackbar";
int low_k = 1;
const int max_k = 5;

// Point MatchingMethod(int, void *);
void computeProbability(BoxLabel &box, Scalar averageBoxColor, double areaBox);

static void onTrackbar(int, void *user)
{
    cv::Mat &src = *(cv::Mat *)user;
    cv::Mat srcCopy = src.clone();
    int k = cv::getTrackbarPos("K trackbar", window_name);
    if (k > 0)
    {
        Mat out = kmeansSegmentation(k, srcCopy);
        imshow(window_name, out);
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string nameFile = "../images/" + std::string(argv[1]);
    cv::Mat in1 = cv::imread(nameFile, CV_32F);
    cv::Mat in2 = cv::imread("../images/Train/Affine_Salad.jpg", CV_32F);

    // img_tm = in1.clone();

    if (!in1.data)
    {
        std::cerr << "ERROR on input image" << std::endl;
        return -1;
    }

    if (!in2.data)
    {
        std::cerr << "ERROR on input image" << std::endl;
        return -1;
    }

    cv::Mat in1_gray;
    cvtColor(in1, in1_gray, cv::COLOR_BGR2GRAY);
    in1_gray.convertTo(in1_gray, CV_8UC1);

    // Hough Transform
    std::vector<cv::Mat> dishes;
    doHough(dishes, in1, in1_gray);

    cv::Mat result = in1.clone();
    std::vector<cv::Rect> mser_boxes;
    for (int d = 0; d < dishes.size(); d++)
    {
        cv::Mat src = dishes[d];
        cv::Mat shifted, bilateral;
        // showImg("dish", src);

        // Pyramidal Filtering with Mean Shift to have a CARTOONISH effect on input image
        cv::pyrMeanShiftFiltering(src, shifted, 25, 30);
        // showImg("Cartoonish (MeanShift Filter)", shifted);

        // Removes the dish from the picture
        removeDish(shifted);

        // Sharpening
        sharpenImg(shifted, SharpnessType::HIGHPASS);

        // Ask for K in dish to compute kMeans
        cv::namedWindow(window_name);
        cv::createTrackbar("K trackbar", window_name, NULL, max_k, onTrackbar, &shifted);
        onTrackbar(1, &shifted);
        cv::waitKey(0);

        // templ_1 = cv::imread("../images/beans.jpg", IMREAD_COLOR);
        // templ_tm.push_back(templ_1);

        // namedWindow(image_window, WINDOW_AUTOSIZE);
        // namedWindow(result_window, WINDOW_AUTOSIZE);
        // Point matchLoc = MatchingMethod(0, 0);

        // Blob detection
        // doMSER(mser_boxes, shifted, result);
        // showImg("MSER", shifted);

        /* for (int i = 0; i < mser_boxes.size(); i++)
        {
            Scalar averageBoxColor = computeAvgColor(shifted, mser_boxes[i]);
            double areaBox = computeArea(mser_boxes[i]);

            cout << averageBoxColor[0] << " -- " << averageBoxColor[1] << " -- " << averageBoxColor[2] << endl;

            double similarityScore;

            // PROBAB.
             //       *Computes probability that it is a food with : -avg color of box -
             //   area of box - similarity score computed by template matching

            BoxLabel b;

            b.mser_box = mser_boxes[i];
            computeProbability(b, averageBoxColor, areaBox);
            string label = enumToString(b.label);
            // showImg("Mser Box " + label, shifted(b.mser_box));
        }
    }


        templ_1 = cv::imread("../images/BeansSIFT.jpg", IMREAD_COLOR);
        // templ_tm.push_back(templ_1);
        // labels.push_back("beans");

        // namedWindow(image_window, WINDOW_AUTOSIZE);
        // namedWindow(result_window, WINDOW_AUTOSIZE);
        // MatchingMethod(0, 0);
        // waitKey(0);

        // SIFT ORB SURF

        Result descriptor = useDescriptor(in1, templ_1, DescriptorType::ORB); // change the descriptor type to use
                                                                              //  SIFT or SURF
        bruteForceHammingSorted(in1, templ_1, descriptor);                    // Hamming for ORB only

        // bruteForceKNN(in1, templ_1, descriptor); // KNN for SIFT and SURF

        /* Segmentation for (int k = 0; k < circles.size(); k++)
        {
            kmeansSegmentation(3, dishes[k]);
        }
        */

        /* SEGMENTATION
         *    cerca in (1) i tipi di cibi (segmentation), sapendo che c'è un solo primo e un solo secondo
         *    e riconosce i cibi tra i 13 del dataset
         */

        /* CONFRONTO FOTO
         *   confronta le due foto per trovare quali cibi sono presenti nella seconda immagine (partendo da quelli della prima)
         */

        /* METRICHE
         *   calcolo delle metriche
         */
    }
    return 0;
}

void computeProbability(BoxLabel &box, Scalar averageBoxColor, double areaBox)
{

    Mat out;
    Scalar upperBound(200, 170, 180); // Lighter Part of Meat
    Scalar lowerBound(95, 80, 60);    // Darker Part of Meat

    bool inRange = true;
    for (int i = 0; i < 3; i++)
    {
        if (averageBoxColor[i] < lowerBound[i] || averageBoxColor[i] > upperBound[i])
        {
            inRange = false;
            break;
        }
    }

    cout << inRange;

    // If area small and color brown --> beans

    // If mean color is dark green --> pasta al pesto

    // If mean color is light green --> salad

    // If area big and color brown --> meat

    // If

    if (areaBox > 2000 && inRange)
    {
        box.label = FoodType::Meat;
    }
    else
    {
        box.label = FoodType::Beans;
    }
}

/*
Point MatchingMethod(int, void *)
{
    Mat img_display;
    img_tm.copyTo(img_display);
    double minVal;
    double maxVal;
    Point minLoc;
    Point maxLoc;
    Point matchLoc;

    for (int i = 0; i < templ_tm.size(); i++)
    {
        int result_cols = img_tm.cols - templ_tm[i].cols + 1;
        int result_rows = img_tm.rows - templ_tm[i].rows + 1;
        result_tm.create(result_rows, result_cols, CV_32FC1);
        matchTemplate(img_tm, templ_tm[i], result_tm, TM_SQDIFF);
        normalize(result_tm, result_tm, 0, 1, NORM_MINMAX, -1, Mat());

        minMaxLoc(result_tm, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
        matchLoc = minLoc;
        // cout << minVal << endl;
        rectangle(img_display, matchLoc, Point(matchLoc.x + templ_tm[i].cols, matchLoc.y + templ_tm[i].rows), Scalar(0, 0, 255), 2, 8, 0);
        rectangle(result_tm, matchLoc, Point(matchLoc.x + templ_tm[i].cols, matchLoc.y + templ_tm[i].rows), Scalar(0, 0, 255), 2, 8, 0);
        cv::putText(img_display, labels[i], cv::Point(matchLoc.x, matchLoc.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        imshow(image_window, img_display);
        imshow(result_window, result_tm);
    }
    return matchLoc;
}
*/
