#include "../headers/utils.h"

bool isInsideCircle(cv::Vec3i circle, int x, int y)
{
    double res = sqrt(pow(circle[0] - x, 2) + pow(circle[1] - y, 2));
    return res > circle[2]; // > radius
}

void showImg(std::string title, cv::Mat image)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, image);
    cv::waitKey();
}

std::string enumToString(FoodType label)
{
    switch (label)
    {
    case FoodType::Meat:
        return "Meat";
    case FoodType::Beans:
        return "Beans";
    default:
        return "UNKNOWN";
    }
}

double computeArea(cv::Rect box)
{
    return box.width * box.height;
}

cv::Mat removeDish(cv::Mat &src)
{
    for (int k = 255; k > 20; k = k - 5)
    {
        cv::Mat mask;
        cv::inRange(src, cv::Scalar(k - 30, k - 30, k - 30), cv::Scalar(k, k, k), mask);
        src.setTo(cv::Scalar(0, 0, 0), mask);
    }
}

/*
    SharpnessType:
    -   LAPLACIAN
    -   HIGHPASS
*/
void sharpenImg(cv::Mat &src, SharpnessType t)
{
    if (t = SharpnessType::HIGHPASS)
    {
        cv::Mat blurred;
        cv::GaussianBlur(src, blurred, cv::Size(7, 7), 3);

        cv::Mat highPass = src - blurred;
        src = src + highPass;
    }
    else if (t = SharpnessType::LAPLACIAN)
    {
        cv::Mat laplacian;
        cv::Laplacian(src, laplacian, CV_16S);
        cv::Mat laplacian8bit;
        laplacian.convertTo(laplacian8bit, CV_8UC3);

        src = src + laplacian8bit;
    }

    // showImg("Sharpened", src);
}

cv::Mat convertGray(cv::Mat &src)
{
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    return gray;
}
