#include "../headers/segmentation.h"

using namespace cv;
using namespace std;

void meanShiftFunct(Mat src)
{
    cv::Mat shifted;
    cv::pyrMeanShiftFiltering(src, shifted, 15, 45);
    imshow("Mean Shifted", shifted);

    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    Mat mask;
    inRange(shifted, Scalar(80, 80, 80), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(255, 255, 255), mask);
    // Show output image
    imshow("Black Background Image", src);

    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1,
                  1, -8, 1,
                  1, 1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    imshow("New Sharped Image", imgResult);

    // Create binary image from source image
    Mat bw;
    cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    threshold(bw, bw, 220, 255, THRESH_OTSU);
    imshow("thershold segmentation", bw);

    waitKey();
}

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