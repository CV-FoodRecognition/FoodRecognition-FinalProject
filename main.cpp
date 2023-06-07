#include <iostream>
#include "headers/segmentation.h"

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

    // print immagini
    /*
        cv::namedWindow("in1");
        cv::imshow("in1", in1);

        cv::namedWindow("in2");
        cv::imshow("in2", in2);

        cv::waitKey(); */

    cv::Mat copy1;
    in1.copyTo(copy1);
    meanShiftFunct(copy1);

    // canny per calcolare edges
    cv::Mat in1_gray;
    CannyThreshold(in1, in1_gray);
    cv::namedWindow("in1");
    cv::imshow("in1", in1);
    cv::waitKey();

    // kmeansSegmentation(15, in1);

    // kmeansSegmentation(15, in2);

    // riduzione rumore, filtraggio, condizioni ottimali per step successivo

    /*
    cv::Mat eqIn1, eqIn2;
    cv::equalizeHist(in1, eqIn1);
    cv::equalizeHist(in2, eqIn2);

    cv::namedWindow("eqIn1");
    cv::imshow("eqIn1", eqIn1);

    cv::namedWindow("eqIn2");
    cv::imshow("eqIn2", eqIn2);

    cv::waitKey(); */

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