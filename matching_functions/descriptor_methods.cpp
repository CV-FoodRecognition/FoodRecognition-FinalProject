#include "include/descriptor_methods.hpp"
#include <iostream>
#include <opencv2/features2d.hpp>

Result useSIFT(cv::Ptr<cv::SIFT> det, cv::Mat &img1, cv::Mat &img2)
{
    Result res;
    det->detectAndCompute(img1, cv::noArray(), res.kp1, res.descriptor1);
    det->detectAndCompute(img2, cv::noArray(), res.kp2, res.descriptor2);

    if (det.empty())
        std::cout << "ERROR - det";
    if (res.kp1.empty())
        std::cout << "ERROR - kp1";
    if (res.kp2.empty())
        std::cout << "ERROR - kp2";
    if (res.descriptor1.empty())
        std::cout << "ERROR - descriptor1";
    if (res.descriptor2.empty())
        std::cout << "ERROR - descriptor2";

    return res;
}
