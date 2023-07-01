#include "descriptor_methods.h"

// Descriptor creation using Factory Pattern
cv::Ptr<cv::Feature2D> createDescriptor(DescriptorType typ)
{
    switch (typ)
    {
    case SURF:
        return cv::xfeatures2d::SURF::create();
    case SIFT:
        return cv::SIFT::create();
    case ORB:
        return cv::ORB::create();
    }
}

Result useDescriptor(cv::Mat &img1, cv::Mat &img2, DescriptorType typ)
{
    cv::Ptr<cv::Feature2D> det = createDescriptor(typ);

    Result res;
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
