#ifndef DESCRIPTOR_METHODS_H
#define DESCRIPTOR_METHODS_H

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "utils.h"

enum DescriptorType
{
    SURF,
    SIFT,
    ORB
};

// Descriptor creation using Factory Pattern
cv::Ptr<cv::Feature2D> createDescriptor(DescriptorType typ);
Result useDescriptor(cv::Mat &img1, cv::Mat &img2, DescriptorType typ);

#endif // DESCRIPTOR_METHODS_H