#include <dirent.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using std::string;
using std::vector;

void analyzeImages(string directory, vector<cv::Mat> &images)
{
    int l;

    if (directory.find("lab") != string::npos)
        l = 13;
    else
        l = 23;

    for (int i = 1; i <= l; i++)
    {
        string filename;
        if (i < 10)
            filename = "i0" + std::to_string(i); // orders the images
        else
            filename = "i" + std::to_string(i); // orders the images

        string filepathBmp = directory + "/" + filename + ".bmp";
        string filepathPng = directory + "/" + filename + ".png";

        cv::Mat image;
        if (FILE *file = fopen(filepathBmp.c_str(), "r"))
        {
            fclose(file);
            image = cv::imread(filepathBmp, cv::IMREAD_COLOR);
        }
        else if (FILE *file = fopen(filepathPng.c_str(), "r"))
        {
            fclose(file);
            image = cv::imread(filepathPng, cv::IMREAD_COLOR);
        }
        else
        {
            std::cout << "ERROR - UTILS.CPP";
            return;
        }
        images.push_back(image);
    }
}