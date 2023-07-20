#include <iostream>
#include "headers/descriptor_methods.h"
#include "headers/matcher_methods.h"
#include "headers/segmentation.h"
#include "headers/utils.h"
#include "headers/ImageProcessor.h"
#include "headers/Leftover.h"
#include "headers/metrics.h"

/*
Written by @nicolacalzone and @rickyvendra
*/

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path> <leftover_path>" << std::endl;
        return -1;
    }

    std::string nameFile1 = "../images/" + std::string(argv[1]);
    std::string nameFile2 = "../images/Leftovers/" + std::string(argv[2]);

    std::vector<cv::Mat> segmentedImages;

    // Read Input
    cv::Mat in1 = cv::imread(nameFile1, CV_32F);
    if (!in1.data)
    {
        std::cerr << "ERROR on input image" << std::endl;
        return -1;
    }
    // Read Leftover
    cv::Mat leftoverImg = cv::imread(nameFile2, cv::IMREAD_COLOR);
    if (!leftoverImg.data)
    {
        std::cerr << "ERROR on leftover input image" << std::endl;
        return -1;
    }

    std::vector<foodTemplate> templates;
    addFood(0, "", "pasta with pesto", 1, "../images/Train/", templates);
    addFood(0, "", "pasta with tomato sauce", 2, "../images/Train/", templates);
    addFood(0, "", "pasta with meat sauce", 3, "../images/Train/", templates);
    addFood(0, "", "pasta with clams and mussels", 4, "../images/Train/", templates); // problems
    addFood(4, "rice", "pilaw rice", 5, "../images/Train/", templates);
    addFood(3, "pork", "grilled pork cutlet", 6, "../images/Train/", templates);
    addFood(2, "fishcutlet", "fish cutlet", 7, "../images/Train/", templates);
    addFood(2, "rabbit", "rabbit", 8, "../images/Train/", templates);
    addFood(2, "seafoodsalad", "seafood salad", 9, "../images/Train/", templates); // problems
    addFood(2, "beans", "beans", 10, "../images/Train/", templates);
    addFood(0, "bread", "bread", 13, "../images/Train/", templates);
    addFood(0, "potatoes", "basil potatoes", 11, "../images/Train/", templates);
    addFood(0, "salad", "salad", 12, "../images/Train/", templates); // problems

    // Hough Transform
    ImageProcessor imgProc;
    imgProc.doHough(in1);
    std::vector<int> &dishesMatches = imgProc.getDishesMatches();
    std::vector<cv::Mat> &removedDishes = imgProc.getDishes();
    // std::vector<cv::Mat> removedDishes = imgProc.removeDish(dishes);
    std::vector<int> &radia1 = imgProc.getRadius();
    std::vector<cv::Vec3f> &acceptedCircles = imgProc.getAcceptedCircles();

    // Hough Transform 2
    ImageProcessor imgProcLeftovers;
    imgProcLeftovers.doHough(leftoverImg);
    std::vector<cv::Mat> &leftovers = imgProcLeftovers.getDishes();
    std::vector<int> &radia2 = imgProcLeftovers.getRadius();

    std::vector<Dish> dishesData;
    for (Mat dish : removedDishes)
    {
        Dish newDish;
        newDish.setDish(dish);
        dishesData.push_back(newDish);
    }
    cv::Mat final = in1.clone();
    std::vector<FoodData> foodData;
    detectAndCompute(in1, removedDishes, dishesMatches, acceptedCircles, foodData, templates, final, dishesData);

    cout << "show final" << endl;
    drawBoundingBoxes(final, foodData);
    // cv::namedWindow("final");
    // cv::imshow("final", final);
    // cv::waitKey();

    for (Dish d : dishesData)
    {
        if (!d.getFoods().empty())
        {
            for (FoodData f : d.getFoods())
            {
                cout << f.label << endl;
                // showImg("segmento cibo", f.segmentArea);
            }
        }
    }

    Leftover leftover;
    leftover.matchLeftovers(removedDishes, dishesData, leftovers, leftoverImg, radia1, radia2, foodData);

    return 0;
}
