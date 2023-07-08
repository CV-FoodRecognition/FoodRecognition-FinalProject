#include "../headers/segmentation.h"

using namespace cv;
using namespace std;

extern std::string window_name;
extern int low_k;
extern cv::Mat src;
extern std::vector<int> kmeans_labels;
extern cv::Mat1f colors;

Scalar computeAvgColor(Mat shifted, Rect box)
{
    Mat roi = shifted(box);

    // Define lower and upper bounds for colors to include
    Scalar lowerb = Scalar(15, 15, 15);
    Scalar upperb = Scalar(240, 240, 240);

    // Create mask to exclude colors outside of bounds
    Mat mask;
    inRange(roi, lowerb, upperb, mask);
    Scalar avg_color = mean(roi, mask);
    Mat image(100, 100, CV_8UC3, avg_color);
    // showImg("average color", image);

    return avg_color;
}

Scalar computeAvgColor(Mat shifted)
{
    // Define lower and upper bounds for colors to include
    Scalar lowerb = Scalar(15, 15, 15);
    Scalar upperb = Scalar(240, 240, 240);

    // Create mask to exclude colors outside of bounds
    Mat mask;
    inRange(shifted, lowerb, upperb, mask);
    Scalar avg_color = mean(shifted, mask);
    Mat image(100, 100, CV_8UC3, avg_color);
    // showImg("average color", image);

    return avg_color;
}

void equalizeHistogram(cv::Mat &src, cv::Mat &dst)
{
    // Convert the image to the lab color space
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // Split the channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);

    // Equalize the histogram of the Y channel
    cv::equalizeHist(labChannels[0], labChannels[0]);

    // Merge the channels back and convert back to BGR color space
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}

void removeBackground(cv::Mat &src)
{
    for (int k = 255; k > 20; k = k - 5)
    {
        Mat mask;
        inRange(src, Scalar(k - 40, k - 40, k - 40), Scalar(k, k, k), mask);
        src.setTo(Scalar(0, 0, 0), mask);
    }
}

void removeShadows(cv::Mat &src, cv::Mat &dst)
{
    // Converting image to LAB color space
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // Splitting the channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);

    // Apply Contrast Limited Adaptive Histogram Equalization
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(1);
    cv::Mat enhancedL;
    clahe->apply(labChannels[0], enhancedL);

    // Merge the channels back and convert back to BGR color space
    enhancedL.copyTo(labChannels[0]);
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}
