#include "headers/segmentation.h"
#include <opencv2/opencv.hpp>

void kmeansSegmentation(int k, cv::Mat &src)
{
    std::vector<int> labels;
    cv::Mat1f colors;
    int attempts = 5;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.);

    cv::Mat input = src.reshape(1, src.rows * src.cols);
    input.convertTo(input, CV_32F);

    cv::kmeans(input, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, colors);

    cv::Mat maskPrimo = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat maskSecondo = cv::Mat::zeros(src.size(), CV_8U);
    cv::Mat maskContorno = cv::Mat::zeros(src.size(), CV_8U);

    for (int i = 0; i < src.rows * src.cols; i++)
    {
        if (labels[i] == 0)
        {
            maskPrimo.at<uchar>(i / src.cols, i % src.cols) = 255;
        }
        else if (labels[i] == 1)
        {
            maskSecondo.at<uchar>(i / src.cols, i % src.cols) = 255;
        }
        else if (labels[i] == 2)
        {
            maskContorno.at<uchar>(i / src.cols, i % src.cols) = 255;
        }
    }

    cv::Mat mask1, mask2, mask3;

    src.copyTo(mask1, maskPrimo);
    src.copyTo(mask2, maskSecondo);
    src.copyTo(mask3, maskContorno);

    cv::namedWindow("mask1");
    cv::imshow("mask1", mask1);
    cv::namedWindow("mask2");
    cv::imshow("mask2", mask2);
    cv::namedWindow("mask3");
    cv::imshow("mask3", mask3);
    cv::waitKey();
}