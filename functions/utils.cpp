#include "../headers/utils.h"

bool isInsideCircle(cv::Vec3i circle, int x, int y)
{
    double res = sqrt(pow(circle[0] - x, 2) + pow(circle[1] - y, 2));
    return res > circle[2]; // > radius
}