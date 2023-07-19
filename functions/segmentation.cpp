#include "../headers/segmentation.h"

/*
Written by @nicolacalzone and @rickyvendra
*/

using namespace cv;
using namespace std;

extern std::string window_name;
extern int low_k;
extern cv::Mat src;
extern std::vector<int> kmeans_labels;
extern cv::Mat1f colors;

cv::Scalar computeAvgColorCIELAB(const cv::Mat &input)
{
    cv::Scalar avg_color = cv::mean(input);
    return avg_color;
}

int boundBreadLeftover(cv::Mat &input, std::vector<cv::Mat> &dishes,
                       cv::Mat &final, std::vector<FoodData> &boundingBoxes)
{
    cv::Mat inClone = input.clone();
    cv::Mat grayDish;

    for (cv::Mat &d : dishes)
    {
        cvtColor(d, grayDish, cv::COLOR_BGR2GRAY);
        cv::Mat mask = grayDish > 0;
        inClone.setTo(cv::Scalar(0, 0, 0), mask);
    }

    cv::Mat inHSV;
    cv::cvtColor(inClone, inHSV, cv::COLOR_BGR2HSV);

    cv::Mat inCIELAB;
    cv::cvtColor(inClone, inCIELAB, cv::COLOR_BGR2Lab);

    Mat mask, mask2;
    cv::inRange(inHSV, Scalar(12, 78, 180), Scalar(25, 110, 240), mask);
    cv::inRange(inHSV, Scalar(12, 30, 70), Scalar(30, 140, 240), mask2);
    // showImg("mask2", mask2);

    for (int i = 0; i < mask.rows; i++)
        for (int j = 0; j < mask.cols; j++)
            mask.at<uchar>(i, j) = 255 - mask.at<uchar>(i, j);

    inHSV.setTo(Scalar(0, 0, 0), mask);

    int kernelSize = 15;
    cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    erode(inHSV, inHSV, kernel);
    kernelSize = 15;
    kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    dilate(inHSV, inHSV, kernel);

    cv::Mat t;
    cv::cvtColor(inHSV, t, COLOR_HSV2BGR);

    cv::Rect box = computeBox(final, t);

    int area = enlargeBoxAndComputeArea(box, mask2, t);

    FoodData bb;
    bb.box = box;

    bb.labels.push_back("bread");
    bb.ids.push_back(13);
    boundingBoxes.push_back(bb);

    return area;
}

int enlargeBoxAndComputeArea(cv::Rect box, cv::Mat inHSV, cv::Mat t)
{
    // Enlarge the box
    int enlargementFactor = 4; // If box is not good adjust this factor
    cv::Rect enlargedBox = box;
    enlargedBox.x -= enlargementFactor * box.width;
    enlargedBox.y -= enlargementFactor * box.height;
    enlargedBox.width += 2 * enlargementFactor * box.width;
    enlargedBox.height += 2 * enlargementFactor * box.height;

    // Enlarged box must be within image boundaries
    enlargedBox.x = std::max(enlargedBox.x, 0);
    enlargedBox.y = std::max(enlargedBox.y, 0);
    enlargedBox.width = std::min(enlargedBox.width, inHSV.cols - enlargedBox.x);
    enlargedBox.height = std::min(enlargedBox.height, inHSV.rows - enlargedBox.y);

    // Compute non-zero pixels within the enlarged box
    cv::Mat roi = t(enlargedBox);
    // showImg("roi", t);
    cv::cvtColor(roi, roi, cv::COLOR_BGR2GRAY);
    int nonZeroArea = cv::countNonZero(roi);

    return nonZeroArea;
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
