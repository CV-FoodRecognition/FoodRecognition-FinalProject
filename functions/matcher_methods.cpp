#include <array>
#include <iostream>
#include "../headers/utils.h"
#include "../headers/matcher_methods.h"

const std::string dir = "../ResultImages/";

void bruteForceHammingSorted(cv::Mat img1, cv::Mat img2, Result res)
{
    // Convert images to 8-bit unsigned integer type
    if (!img1.empty() && img1.type() != CV_8U)
        img1.convertTo(img1, CV_8U);
    if (!img2.empty() && img2.type() != CV_8U)
        img2.convertTo(img2, CV_8U);

    // Check and convert descriptors to 8-bit unsigned integer type
    if (res.descriptor1.type() != CV_8U)
        res.descriptor1.convertTo(res.descriptor1, CV_8U);
    if (res.descriptor2.type() != CV_8U)
        res.descriptor2.convertTo(res.descriptor2, CV_8U);

    cv::BFMatcher bf(cv::NORM_HAMMING, true);

    std::vector<cv::DMatch> matches;
    bf.match(res.descriptor1, res.descriptor2, matches);

    std::sort(matches.begin(), matches.end());

    cv::Mat imgMatches;
    cv::drawMatches(img1, res.kp1, img2, res.kp2, matches, imgMatches);
    // imgMatches = checkTransformRotation(img1, img2, matches, res);

    std::string file = "Hamming - ORB Sorted.png";

    showImg(file, imgMatches);
}

void bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res)
{
    // Convert images to 8-bit unsigned integer type
    img1.convertTo(img1, CV_8U);
    img2.convertTo(img2, CV_8U);

    cv::BFMatcher bf;
    std::vector<std::vector<cv::DMatch>> matches;
    bf.knnMatch(res.descriptor1, res.descriptor2, matches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (const auto &match : matches)
    {
        if (match[0].distance < 0.6 * match[1].distance)
        {
            goodMatches.push_back(match[0]);
        }
    }

    cv::Mat imgMatches;
    cv::drawMatches(img1, res.kp1, img2, res.kp2, matches, imgMatches);

    std::string file = "KNN - Matching - SURF.png";

    showImg(file, imgMatches);
}