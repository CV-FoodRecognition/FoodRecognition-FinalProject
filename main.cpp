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

    if (!in1.data)
    {
        std::cerr << "ERROR" << std::endl;
        return -1;
    }

    if (!in2.data)
    {
        std::cerr << "ERROR" << std::endl;
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

        // Pyramidal Filtering with Mean Shift to have a CARTOONISH effect on input image
        cv::pyrMeanShiftFiltering(src, shifted, 25, 30);
        showImg("Cartoonish (MeanShift Filter)", shifted);

        // Removes the dish from the picture
        removeDish(shifted);

        // Sharpening
        sharpenImg(shifted, SharpnessType::HIGHPASS);

        // Blob detection
        doMSER(mser_boxes, shifted, result);
        showImg("MSER", shifted);

        for (int i = 0; i < mser_boxes.size(); i++)
        {
            Scalar averageBoxColor = computeAvgColor(shifted, mser_boxes[i]);
            double areaBox = computeArea(mser_boxes[i]);
            double similarityScore;

            /*      PROBAB.
             *   Computes probability that it is a food with:
                    - avg color of box
                    - area of box
                    - similarity score computed by template matching
             */
        }
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
