#include <iostream>
#include "headers/segmentation.h"
#include "headers/utils.h"

using cv::Mat;
using std::string;
using std::vector;

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
    cv::Mat in2 = cv::imread("../images/leftover1.jpg", CV_32F);

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

    cv::Mat contoursCircles = cv::Mat::zeros(in1.size(), CV_8UC1);

    for (size_t i = 0; i < circles.size(); i++)
    {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]); // c0 = x coord , c1 = y coord of the circle
        int radius = c[2];                        // c2 = ray of the circle
        circle(contoursCircles, center, radius, 255, 1);
    }

    cv::namedWindow("in1_gray");
    cv::imshow("in1_gray", contoursCircles);

    vector<vector<cv::Point>> contours;

    for (int k = 0; k < circles.size(); k++)
    {
        cv::Mat dish = cv::Mat::zeros(in1.size(), CV_8UC3);
        cv::findContours(contoursCircles, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(in1, contours, -1, 255, 1, 8);
        cv::namedWindow("print");
        cv::imshow("print", in1);

        cv::Vec3i c = circles[k];

        for (int i = 0; i < in1.rows; ++i)
        {
            for (int j = 0; j < in1.cols; ++j)
            {
                if (isInsideCircle(c, i, j)) // outside of the circle
                {
                    dish.at<unsigned char>(i, j) = 0;
                }
                else // inside of the circle
                {
                    dish.at<unsigned char>(i, j) = in1.at<unsigned char>(j, i);
                }
            }
        }
        cv::namedWindow("dish");
        cv::imshow("dish", dish);
        cv::waitKey();
    }

    // cerca in (1) i tipi di cibi (segmentation), sapendo che c'è un solo primo e un solo secondo
    // e riconosce i cibi tra i 13 del dataset

    // confronta le due foto per trovare quali cibi sono presenti nella seconda immagine (partendo da quelli della prima)

    // calcolo delle metriche

    return 0;
}

// cerca_cibi(){
// cerca cibo 1: pasta con pesto: cercare un piatto e riconoscere verde
// cerca cibo...
// cerca vibo 13

// return cibi trovati
//}