#ifndef DESCRIPTOR_METHODS_H
#define DESCRIPTOR_METHODS_H

#include <opencv2/core.hpp>
#include "utils.h"

enum DescriptorType
{
    SURF,
    SIFT,
    ORB
};

// Descriptor creation using Factory Pattern
cv::Ptr<cv::Feature2D> createDescriptor(DescriptorType type);
Result useDescriptor(cv::Mat &img1, cv::Mat &img2, DescriptorType type);

#endif // DESCRIPTOR_METHODS_H