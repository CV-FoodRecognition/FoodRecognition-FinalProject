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

Scalar computeAvgColorHSV(Mat img)
{
    Mat input = img.clone();
    // showImg("dish", input);

    Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    Mat mask_yellow;
    inRange(hsv, Scalar(14, 0, 0), Scalar(25, 255, 255), mask_yellow);
    // for (int i = 0; i < mask_yellow.rows; i++)
    //{
    //     for (int j = 0; j < mask_yellow.cols; j++)
    //     {
    //         mask_yellow.at<uchar>(i, j) = 255 - mask_yellow.at<uchar>(i, j);
    //     }
    // }
    hsv.setTo(Scalar(0, 0, 0), mask_yellow);
    cv::cvtColor(hsv, input, cv::COLOR_HSV2BGR);

    // showImg("noYellow", input);

    // Define lower and upper bounds for colors to include
    Scalar lowerb = Scalar(25, 0, 0);
    Scalar upperb = Scalar(255, 255, 255);
    // Create mask to exclude colors outside of bounds
    Mat mask;
    inRange(input, lowerb, upperb, mask);
    Scalar avg_color = mean(input, mask);

    Mat color(100, 100, CV_8UC3, avg_color);
    // showImg("average color", color);

    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);
    Scalar hsv_color = hsv.at<uchar>(0, 0);
    cout << "hue: " << hsv_color[0] << endl;

    return hsv_color;
}

int computeBestDish(foodTemplate food, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches)
{
    int best_dish_id = -1;
    for (int d = 0; d < dishesMatches.size(); d++)
    {
        Scalar avgColor = computeAvgColorHSV(dishes[d]);

        if (food.id == 1)
        {
            if (avgColor[0] > 30 && avgColor[0] < 50)
            {
                return d;
            }
        }
        if (food.id == 2)
        {
            if (avgColor[0] > 0 && avgColor[0] < 11)
            {
                best_dish_id = d;
                return d;
            }
        }
        if (dishesMatches[d] > dishesMatches[best_dish_id])
        {
            best_dish_id = d;
        }
    }
    return best_dish_id;
}

void boundBread(cv::Mat &input, std::vector<cv::Mat> &dishes,
                cv::Mat &final, std::vector<BoundingBox> &boundingBoxes)
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

    Mat mask;
    cv::inRange(inHSV, Scalar(12, 80, 140), Scalar(28, 230, 255), mask);
    for (int i = 0; i < mask.rows; i++)
        for (int j = 0; j < mask.cols; j++)
            mask.at<uchar>(i, j) = 255 - mask.at<uchar>(i, j);

    inHSV.setTo(Scalar(0, 0, 0), mask);

    int kernelSize = 3;
    cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    erode(inHSV, inHSV, kernel);
    kernelSize = 7;
    kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    dilate(inHSV, inHSV, kernel);

    cv::Mat t;
    cv::cvtColor(inHSV, t, COLOR_HSV2BGR);

    cv::Rect box = computeBox(final, t);

    BoundingBox bb;
    bb.box = box;
    bb.label = "bread";
    boundingBoxes.push_back(bb);
}

void boundPasta(cv::Mat &dish, cv::Mat &final, std::string label,
                std::vector<int> forbidden, int max_key, std::vector<BoundingBox> &boundingBoxes)
{
    bool allowed = true;
    cv::Rect box;
    for (int i = 0; i < forbidden.size(); i++)
    {
        if (forbidden[i] == max_key)
        {
            allowed = false;
            break;
        }
    }
    if (allowed)
    {

        box = computeBox(final, dish);
        BoundingBox bb;
        bb.box = box;
        bb.label = label;
        boundingBoxes.push_back(bb);
    }
    return;
}

void drawBoundingBoxes(cv::Mat &final, std::vector<BoundingBox> &boundingBoxes)
{
    for (BoundingBox &bb : boundingBoxes)
    {
        cv::putText(final, bb.label, cv::Point(bb.box.x, bb.box.y - 20), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255), 2, 8, false);
        cv::rectangle(final, bb.box, CV_RGB(255, 0, 0), 2);
    }
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
