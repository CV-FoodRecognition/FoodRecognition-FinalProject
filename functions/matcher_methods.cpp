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

int bruteForceKNN(cv::Mat img1, cv::Mat img2, Result res, cv::Mat &final)
{
    std::cout << "inizo\n";

    img1.convertTo(img1, CV_8U);
    img2.convertTo(img2, CV_8U);

    std::cout << "conversion images\n";

    cv::BFMatcher bf;
    std::vector<std::vector<cv::DMatch>> matches;
    bf.knnMatch(res.descriptor1, res.descriptor2, matches, 2);

    std::cout << "knnMatches\n";

    std::vector<cv::DMatch> goodMatches;
    for (const auto &match : matches)
    {
        if (match[0].distance < 0.7 * match[1].distance)
        {
            goodMatches.push_back(match[0]);
        }
    }

    std::cout << "goodMatches";

    cv::Mat imgMatches;
    cv::drawMatches(img1, res.kp1, img2, res.kp2, goodMatches, imgMatches);

    std::cout << "draw matches\n";

    int x = final.cols;
    int y = final.rows;
    int max_x = 0;
    int max_y = 0;
    for (int i = 0; i < goodMatches.size(); i++)
    {
        int id = goodMatches[i].queryIdx;
        float kp_x = res.kp1[id].pt.x;
        float kp_y = res.kp1[id].pt.y;
        std::cout << "kp_x:  " << kp_x << std::endl;
        std::cout << "kp_y:  " << kp_y << std::endl;
        if (kp_x < x)
        {
            x = cvRound(kp_x);
        }
        if (kp_x > max_x)
        {
            max_x = cvRound(kp_x);
        }
        if (kp_y < y)
        {
            y = cvRound(kp_y);
        }
        if (kp_y > max_y)
        {
            max_y = cvRound(kp_y);
        }
        std::cout << "x:  " << x << std::endl;
        std::cout << "y:  " << y << std::endl;
        std::cout << "max_x:  " << max_x << std::endl;
        std::cout << "max_y:  " << max_y << std::endl;
    }

    cv::Rect boundingBox(x, y, max_x - x, max_y - y);
    cv::rectangle(final, boundingBox, CV_RGB(0, 255, 0));

    // SAVING RESULT
    std::string file = "KNN - Matching - SIFT.png";
    showImg(file, imgMatches);

    return goodMatches.size();
}
