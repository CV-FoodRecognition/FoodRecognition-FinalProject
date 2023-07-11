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
    cv::waitKey(0);
}

void concatShowImg(std::string title, cv::Mat image1, cv::Mat image2)
{
    cv::Mat combined;
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::hconcat(image1, image2, combined);
    cv::imshow(title, combined);
    cv::waitKey(0);
}

void addFood(int size, std::string fileName, std::string label, int id,
             std::string path, std::vector<foodTemplate> &templates)
{
    cv::Mat temp_template;
    foodTemplate myFood;
    for (int i = 1; i <= size; i++)
    {
        std::string file = path + fileName + std::to_string(i) + ".jpg";
        temp_template = cv::imread(file, cv::IMREAD_GRAYSCALE);
        myFood.foodTemplates.push_back(temp_template);
        myFood.label = label;
        myFood.id = id;
    }
    templates.push_back(myFood);
    return;
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

double computeCircleArea(double radius)
{
    return M_PI * radius * radius;
}

/*
    Computes area of a KMeans Segment
*/
void computeSegmentArea(SegmentAreas &sa)
{
    cv::Mat maskYellow, maskBlue, maskGreen, maskRed, maskBlack;
    inRange(sa.p1, cv::Scalar(0, 254, 254), cv::Scalar(0, 255, 255), maskYellow);
    inRange(sa.p1, cv::Scalar(254, 0, 0), cv::Scalar(255, 0, 0), maskBlue);
    inRange(sa.p1, cv::Scalar(0, 254, 0), cv::Scalar(0, 255, 0), maskGreen);
    inRange(sa.p1, cv::Scalar(0, 0, 254), cv::Scalar(0, 0, 255), maskRed);
    inRange(sa.p1, cv::Scalar(0, 0, 0), cv::Scalar(10, 10, 10), maskBlack);

    sa.areaYellow = countNonZero(maskYellow);
    sa.areaBlue = countNonZero(maskBlue);
    sa.areaGreen = countNonZero(maskGreen);
    sa.areaRed = countNonZero(maskRed);
    sa.areaBlack = countNonZero(maskBlack);
}

void removeDish(cv::Mat &src)
{
    for (int k = 255; k > 20; k = k - 5)
    {
        cv::Mat mask;
        cv::inRange(src, cv::Scalar(k - 30, k - 30, k - 30), cv::Scalar(k, k, k), mask);
        src.setTo(cv::Scalar(0, 0, 0), mask);
    }
}

void acceptCircles(cv::Mat &in, cv::Mat &mask, cv::Mat &temp,
                   cv::Vec3i &c, cv::Point &center, int radius,
                   std::vector<cv::Vec3f> &accepted_circles,
                   std::vector<int> &dishesMatches, std::vector<cv::Mat> &dishes)
{
    if (accepted_circles.size() == 0)
    {
        dishesMatches.push_back(0);
        cv::circle(mask, center, radius, 255, -1);
        cv::Mat mask_colored;
        in.copyTo(mask_colored, mask);
        removeDish(mask_colored);
        dishes.push_back(mask_colored);
        accepted_circles.push_back(c);
        cv::circle(temp, center, 1, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        cv::circle(temp, center, radius, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }
    if (!isInside(accepted_circles, center))
    {
        dishesMatches.push_back(0);
        cv::circle(mask, center, radius, 255, -1);
        cv::Mat mask_colored;
        in.copyTo(mask_colored, mask);
        removeDish(mask_colored);
        dishes.push_back(mask_colored);
        accepted_circles.push_back(c);
        cv::circle(temp, center, 1, cv::Scalar(255, 0, 0), 3, cv::LINE_AA);
        cv::circle(temp, center, radius, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
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
