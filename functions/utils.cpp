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

/*
    SharpnessType:
    -   LAPLACIAN
    -   HIGHPASS
*/
cv::Mat sharpenImg(cv::Mat src, SharpnessType t)
{
    cv::Mat sharpenedImg;

    if (t = SharpnessType::HIGHPASS)
    {
        cv::Mat blurred;
        cv::GaussianBlur(src, blurred, cv::Size(7, 7), 3);

        cv::Mat highPass = src - blurred;
        sharpenedImg = src + highPass;
    }
    else if (t = SharpnessType::LAPLACIAN)
    {
        cv::Mat laplacian;
        cv::Laplacian(src, laplacian, CV_16S);
        cv::Mat laplacian8bit;
        laplacian.convertTo(laplacian8bit, CV_8UC3);
        // showImg("laplacian", src);

        sharpenedImg = src + laplacian8bit;
    }

    showImg("Sharpened", sharpenedImg);
    return sharpenedImg;
}