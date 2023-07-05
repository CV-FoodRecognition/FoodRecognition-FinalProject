#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImagePreprocessor.hpp"
using std::cout;
using std::endl;
using namespace cv;

int main(int argc, char *argv[])
{

    std::vector<cv::Mat> templates;
    std::vector<std::string> labels;
    Mat templ_fagioli = cv::imread("../images/Train/fagioli.jpg", cv::IMREAD_GRAYSCALE);
    Mat templ_conchiglie = cv::imread("../images/Train/conchiglie.jpg", cv::IMREAD_GRAYSCALE);
    templates.push_back(templ_fagioli);
    labels.push_back("fagioli");
    templates.push_back(templ_conchiglie);
    labels.push_back("pasta");
    Result result;

    cv::GaussianBlur(in1_gray, in1_gray, cv::Size(7, 7), 1.5, 1.5, 4);
    // Hough Circles per ottenere solo i piatti
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(in1_gray, circles, cv::HOUGH_GRADIENT,
                     1, 25, 100, 100,
                     150, 500); // min radius and max radius

    std::vector<cv::Mat> dishes;
    std::vector<int> dishesMatches;
    for (int k = 0; k < circles.size(); k++)
    {
        cv::Mat mask = cv::Mat::zeros(in1.size(), CV_8UC1);
        cv::Mat dish = cv::Mat::zeros(in1.size(), CV_8UC3);
        dishes.push_back(dish);
        dishesMatches.push_back(0);
        cv::Vec3i c = circles[k];
        cv::Point center = cv::Point(c[0], c[1]); // c0 = x coord , c1 = y coord of the circle
        int radius = c[2];                        // c2 = ray of the circle
        cv::circle(mask, center, radius, 255, -1);

        in1.copyTo(dishes[k], mask);
    }
    Mat final = in1.clone();
    for (int t = 0; t < templates.size(); t++)
    {
        cout << "inizio a cercare match" << endl;
        for (int i = 0; i < dishes.size(); i++)
        {
            result = useDescriptor(dishes[i], templates[t], DescriptorType::SIFT);
            dishesMatches[i] = bruteForceKNN(dishes[i], templates[t], result, Mat(in1.rows, in1.cols, CV_8UC3));
            std::cout << "matches dishes " << i << " : " << dishesMatches[i] << std::endl;
        }
        cout << "finito di cercare i match" << endl;
        int max_key = 0;
        for (int i = 0; i < dishesMatches.size(); i++)
        {
            if (dishesMatches[i] > dishesMatches[max_key])
            {
                max_key = i;
            }
        }
        cout << "trovato il piatto con più match" << endl;
        if (dishesMatches[max_key] > 0)
        {
            cout << "match finale (quello più giusto)" << endl;
            result = useDescriptor(dishes[i], templates[t], DescriptorType::SIFT);
            bruteForceKNN(in1, templates[t], result, final);
        }
    }

    cout << "show final" << endl;

    cv::namedWindow("final");
    cv::imshow("final", final);
    cv::waitKey();
    return 0;
}