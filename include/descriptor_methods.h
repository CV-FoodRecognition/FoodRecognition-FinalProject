#ifndef METHODS_H
#define METHODS_H

#include <opencv2/features2d.hpp>
#include "utils.hpp"

Result useSIFT(cv::Ptr<cv::SIFT> det,
               cv::Mat &img1,
               cv::Mat &img2);

#endif // METHODS_H