#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"

using cv::Vec3b, cv::Rect, cv::Scalar, cv::Point, cv::Mat;
using std::string, std::vector;

// GLOBAL VARIABLES
std::string window_name = "Edge Map";
const int max_lowThreshold = 70;
int lowThreshold = 0;
const int kernel_size = 7;
cv::Mat dst, detected_edges;

static void CannyThreshold(cv::Mat &in1, cv::Mat &in1_gray)
{
    cv::cvtColor(in1, in1_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(in1_gray, dst, cv::Size(kernel_size, kernel_size), 0, 0);
    cv::Canny(dst, detected_edges, lowThreshold, max_lowThreshold);
    dst = cv::Scalar::all(0);
    in1.copyTo(dst, detected_edges);
    cv::imshow(window_name, dst);
}

int main(int argc, char **argv)
{
    // riceve in input 2 foto: vassoio prima (1) e dopo pranzo (2)
    cv::Mat in1 = cv::imread("../images/food_image.jpg", CV_32F);
    cv::Mat in2 = cv::imread("../images/Train/Affine_Salad.jpg", CV_32F);

    if (in1.data == NULL)
    {
        std::cout << "ERROR" << std::endl;
        return -1;
    }

    if (in2.data == NULL)
    {
        std::cout << "ERROR" << std::endl;
        return -1;
    }

    cv::Mat in1_gray;
    cvtColor(in1, in1_gray, cv::COLOR_BGR2GRAY);
    in1_gray.convertTo(in1_gray, CV_8UC1);

    cv::GaussianBlur(in1_gray, in1_gray, cv::Size(7, 7), 1.5, 1.5, 4);
    // Hough Circles per ottenere solo i piatti
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(in1_gray, circles, cv::HOUGH_GRADIENT,
                     1, 40, 100, 100,
                     150, 400); // min radius and max radius

    std::vector<cv::Mat> dishes;
    for (int k = 0; k < circles.size(); k++)
    {
        cv::Mat mask = cv::Mat::zeros(in1.size(), CV_8UC1);
        cv::Mat dish = cv::Mat::zeros(in1.size(), CV_8UC3);
        dishes.push_back(dish);
        cv::Vec3i c = circles[k];
        cv::Point center = cv::Point(c[0], c[1]); // c0 = x coord , c1 = y coord of the circle
        int radius = c[2];                        // c2 = ray of the circle
        circle(mask, center, radius, 255, -1);
        in1.copyTo(dishes[k], mask);

        // showImg("dishes", dishes[k]);
    }

    cv::Mat result = in1.clone();

    for (int d = 0; d < dishes.size(); d++)
    {

        cv::Mat src = dishes[d];
        cv::Mat shifted;
        cv::Mat bilateral;
        cv::pyrMeanShiftFiltering(src, shifted, 25, 30);
        imshow("Mean Shifted", shifted);

        for (int k = 255; k > 20; k = k - 5)
        {
            cv::Mat mask;
            cv::inRange(shifted, Scalar(k - 40, k - 40, k - 40), Scalar(k, k, k), mask);
            shifted.setTo(Scalar(0, 0, 0), mask);
        }

        // sharpening
        shifted = sharpenImg(shifted, SharpnessType::HIGHPASS);

        cv::Mat hsvImage;
        cvtColor(shifted, hsvImage, cv::COLOR_BGR2HSV);
        cv::Mat mask;

        cv::inRange(hsvImage, cv::Scalar(35, 50, 50), cv::Scalar(75, 255, 255), mask);
        const cv::Scalar kAssignedScalar = cv::Scalar(120, 100, 100);
        hsvImage.setTo(kAssignedScalar, mask);
        cv::cvtColor(hsvImage, shifted, cv::COLOR_HSV2BGR);
        showImg("hsvImage", hsvImage);

        cv::Ptr<cv::MSER> ms = cv::MSER::create();
        std::vector<std::vector<cv::Point>> regions;
        std::vector<cv::Rect> mser_bbox;
        ms->detectRegions(shifted, regions, mser_bbox);

        for (int i = 0; i < regions.size(); i++)
        {
            cv::rectangle(shifted, mser_bbox[i], CV_RGB(0, 255, 0));

            cv::Mat mask, bg, fg;

            // grabCut(shifted, mask, mser_bbox[i], bg, fg, 1, GC_INIT_WITH_RECT);
            Rect rect = mser_bbox[i];
            int area = rect.width * rect.height;
            for (int i = rect.x; i < rect.x + rect.width; i++)
            {
                for (int j = rect.y; j < rect.y + rect.height; j++)
                {
                    result.at<Vec3b>(Point(i, j))[0] = 0;
                    result.at<Vec3b>(Point(i, j))[1] = 0;
                    result.at<Vec3b>(Point(i, j))[2] = 255;
                }
            }
        }
        showImg("MSER", shifted);
    }

    /* SIFT ORB SURF

    Result descriptor = useDescriptor(in1, in2, DescriptorType::ORB); // change the descriptor type to use
                                                                               //  SIFT or SURF

    bruteForceHammingSorted(in1, in2, descriptor);      // Hamming for ORB only

    // bruteForceKNN(in1, in2, descriptor)              // KNN for SIFT and SURF


    // Segmentation
    for (int k = 0; k < circles.size(); k++)
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

    return 0;
}

// cerca_cibi(){
// cerca cibo 1: pasta con pesto: cercare un piatto e riconoscere verde
// cerca cibo...
// cerca cibo 13

// return cibi trovati
//}