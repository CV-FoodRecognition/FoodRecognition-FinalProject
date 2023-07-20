#include "../headers/compute_dish.h"

using namespace cv;
using namespace std;

cv::Scalar computeAvgColorHSV(cv::Mat img)
{
    cv::Mat input = img.clone();
    // imshow("dish", input);
    // waitKey();
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask_yellow;
    cv::inRange(hsv, cv::Scalar(14, 0, 0), cv::Scalar(25, 255, 255), mask_yellow);
    // for (int i = 0; i < mask_yellow.rows; i++)
    //{
    //     for (int j = 0; j < mask_yellow.cols; j++)
    //     {
    //         mask_yellow.at<uchar>(i, j) = 255 - mask_yellow.at<uchar>(i, j);
    //     }
    // }
    hsv.setTo(cv::Scalar(0, 0, 0), mask_yellow);
    cv::cvtColor(hsv, input, cv::COLOR_HSV2BGR);
    // imshow("no yellow", input);
    // waitKey();

    // Define lower and upper bounds for colors to include
    cv::Scalar lowerb = cv::Scalar(25, 0, 0);
    cv::Scalar upperb = cv::Scalar(255, 255, 255);
    // Create mask to exclude colors outside of bounds
    cv::Mat mask;
    cv::inRange(input, lowerb, upperb, mask);
    cv::Scalar avg_color = mean(input, mask);

    cv::Mat color(100, 100, CV_8UC3, avg_color);
    // imshow("average color", color);
    // waitKey();

    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar hsv_color = hsv.at<cv::Vec3b>(1, 1);

    return hsv_color;
}

cv::Rect computeBox(cv::Mat &dish)
{
    int x = dish.cols;
    int y = dish.rows;
    int max_x = 0;
    int max_y = 0;
    for (int i = 0; i < dish.cols; i++)
    {
        for (int j = 0; j < dish.rows; j++)
        {
            if (dish.at<cv::Vec3b>(cv::Point(i, j))[0] != 0 &&
                dish.at<cv::Vec3b>(cv::Point(i, j))[1] != 0 &&
                dish.at<cv::Vec3b>(cv::Point(i, j))[2] != 0)
            {
                if (i < x)
                {
                    x = i;
                }
                if (i > max_x)
                {
                    max_x = i;
                }
                if (j < y)
                {
                    y = j;
                }
                if (j > max_y)
                {
                    max_y = j;
                }
            }
        }
    }
    return cv::Rect(x, y, max_x - x, max_y - y);
}

cv::Mat getCorrectSegment(cv::Mat &dish, foodTemplate &foodTemplate, cv::Mat &maskYellow, cv::Mat &maskBlue)
{
    cv::Mat food1;
    int matchesCount1 = 0;
    dish.copyTo(food1, maskYellow);
    cv::Mat food2;
    dish.copyTo(food2, maskBlue);
    int matchesCount2 = 0;
    cv::BFMatcher bf;
    std::vector<std::vector<cv::DMatch>> matches1;
    std::vector<std::vector<cv::DMatch>> matches2;
    for (int t = 0; t < foodTemplate.foodTemplates.size(); t++)
    {
        cv::Mat img2 = foodTemplate.foodTemplates[t];
        img2.convertTo(img2, CV_8U);

        Result res1 = useDescriptor(food1, foodTemplate.foodTemplates[t], DescriptorType::SIFT);
        Result res2 = useDescriptor(food2, foodTemplate.foodTemplates[t], DescriptorType::SIFT);

        bf.knnMatch(res1.descriptor1, res1.descriptor2, matches1, 2);
        bf.knnMatch(res2.descriptor1, res2.descriptor2, matches2, 2);
        for (const auto &match : matches1)
        {
            if (match[0].distance < 0.62 * match[1].distance)
            {
                matchesCount1++;
            }
        }
        for (const auto &match : matches2)
        {
            if (match[0].distance < 0.62 * match[1].distance)
            {
                matchesCount2++;
            }
        }
    }

    if (matchesCount1 > matchesCount2)
    {
        return maskYellow;
    }
    else
    {
        return maskBlue;
    }
}

void detectAndCompute(cv::Mat in1, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches,
                      std::vector<cv::Vec3f> accepted_circles, std::vector<FoodData> &foodData, std::vector<foodTemplate> templates,
                      cv::Mat &final, std::vector<Dish> &dishesData)
{
    Result resSIFT;
    std::vector<int> forbidden;

    bool alreadyFoundFirstDish = false;
    bool alreadyFoundSecondDish = false;
    for (int f = 0; f < templates.size(); f++)
    { // for every food
        for (int d = 0; d < dishes.size(); d++)
        {
            dishesMatches[d] = 0;
        }
        for (int t = 0; t < templates[f].foodTemplates.size(); t++)
        { // for every food template
            for (int d = 0; d < dishes.size(); d++)
            { // look for matches in every dish
                resSIFT = useDescriptor(dishes[d], templates[f].foodTemplates[t], DescriptorType::SIFT);
                dishesMatches[d] = dishesMatches[d] + bruteForceKNN(dishes[d], templates[f].foodTemplates[t], resSIFT);
            }
        }
        int max_key = 0;
        cv::Scalar avgColor;
        if (templates[f].id != 13)
        {
            max_key = computeBestDish(templates[f], dishes, dishesMatches);
            if (max_key > -1)
            {
                avgColor = computeAvgColor(dishes[max_key]);
            }
        }

        if (max_key > -1)
        {
            if (templates[f].id == 1 && avgColor[0] > 30 && avgColor[0] < 50 && !alreadyFoundFirstDish)
            {
                // pesto
                boundPasta(dishes[max_key], final, templates[f].label, templates[f].id, forbidden, max_key, foodData);
                forbidden.push_back(max_key);
                FoodData bb;
                bb.label = templates[f].label;
                bb.id = templates[f].id;
                bb.box = computeBox(final, dishes[max_key]);
                Mat segment = Mat::zeros(final.size(), CV_8U);
                for (int i = 0; i < final.rows; i++)
                    for (int j = 0; j < final.cols; j++)
                        if (dishes[max_key].at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0))
                        {
                            segment.at<uchar>(i, j) = 255;
                        }

                bb.segmentArea = segment;
                dishesData[max_key].addFoods(bb);

                alreadyFoundFirstDish = true;
            }
            else if ((templates[f].id == 2 || templates[f].id == 3) && avgColor[0] > 0 && avgColor[0] < 11 && avgColor[1] > 150 && !alreadyFoundFirstDish)
            {
                // tomato
                boundPasta(dishes[max_key], final, templates[f].label, templates[f].id, forbidden, max_key, foodData);
                forbidden.push_back(max_key);

                FoodData bb;
                bb.label = templates[f].label;
                bb.id = templates[f].id;
                bb.box = computeBox(final, dishes[max_key]);
                Mat segment = Mat::zeros(final.size(), CV_8U);
                for (int i = 0; i < final.rows; i++)
                    for (int j = 0; j < final.cols; j++)
                        if (dishes[max_key].at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0))
                        {
                            segment.at<uchar>(i, j) = 255;
                        }

                bb.segmentArea = segment;
                dishesData[max_key].addFoods(bb);

                alreadyFoundFirstDish = true;
            }
            else if (templates[f].id == 4 && avgColor[0] > 10 && avgColor[0] < 20 && avgColor[1] > 140 && !alreadyFoundFirstDish)
            {
                // pesto
                boundPasta(dishes[max_key], final, templates[f].label, templates[f].id, forbidden, max_key, foodData);
                forbidden.push_back(max_key);

                FoodData bb;
                bb.label = templates[f].label;
                bb.id = templates[f].id;
                bb.box = computeBox(final, dishes[max_key]);
                Mat segment = Mat::zeros(final.size(), CV_8U);
                for (int i = 0; i < final.rows; i++)
                    for (int j = 0; j < final.cols; j++)
                        if (dishes[max_key].at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0))
                        {
                            segment.at<uchar>(i, j) = 255;
                        }

                bb.segmentArea = segment;
                dishesData[max_key].addFoods(bb);
                alreadyFoundFirstDish = true;
            }
            else if (templates[f].id == 11)
            {
                // potatoes
                boundPotatoes(dishes[max_key], final, foodData, forbidden, max_key, dishesData);
            }
            else if (templates[f].id > 4)
            {
                if (templates[f].id == 5 && !alreadyFoundFirstDish)
                {
                    bruteForceKNN(in1, templates[f], dishes[max_key], final, foodData, max_key, dishesData);
                    forbidden.push_back(max_key);
                    alreadyFoundFirstDish = true;
                }
                else
                {
                    if (templates[f].id == 6 || templates[f].id == 7 || templates[f].id == 8 || templates[f].id == 9)
                    {
                        if (!alreadyFoundSecondDish)
                        {
                            bruteForceKNN(in1, templates[f], dishes[max_key], final, foodData, max_key, dishesData);
                            alreadyFoundSecondDish = true;
                        }
                    }
                    else
                    {
                        bruteForceKNN(in1, templates[f], dishes[max_key], final, foodData, max_key, dishesData);
                    }
                }
            }
        }
        if (dishes.size() > 2 && templates[f].id == 12)
        {
            // salad
            boundSalad(in1, accepted_circles, final, foodData, dishes, dishesData);
        }
        else if (templates[f].id == 13)
        {
            // bread
            boundBread(in1, dishes, final, foodData);
        }
    }
}

int computeBestDish(foodTemplate food, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches)
{
    int best_dish_id = -1;
    for (int d = 0; d < dishesMatches.size(); d++)
    {
        cv::Scalar avgColor = computeAvgColor(dishes[d]);
        if (food.id == 1)
        {
            if (avgColor[0] > 30 && avgColor[0] < 50)
            {
                return d;
            }
        }
        else if (food.id == 2 || food.id == 3)
        {
            if (avgColor[0] > 0 && avgColor[0] < 11 && avgColor[1] > 150)
            {
                return d;
            }
        }
        else if (food.id == 4)
        {
            if (avgColor[0] > 10 && avgColor[0] < 20 && avgColor[1] > 140)
            {
                return d;
            }
        }
        else if (food.id == 11)
        {
            avgColor = computeAvgColor(dishes[d], 11);
            if (avgColor[0] > 20 && avgColor[0] < 30 && avgColor[1] < 155 && avgColor[2] > 180)
            {
                best_dish_id = d;
                return d;
            }
        }
        else if (dishesMatches[d] > dishesMatches[best_dish_id] && dishesMatches[d] > 5)
        {
            best_dish_id = d;
        }
    }
    return best_dish_id;
}

cv::Scalar computeAvgColor(cv::Mat img)
{
    cv::Mat input = img.clone();
    // imshow("dish", input);
    // waitKey();
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask_yellow;
    cv::inRange(hsv, cv::Scalar(14, 0, 0), Scalar(25, 255, 255), mask_yellow);
    // for (int i = 0; i < mask_yellow.rows; i++)
    //{
    //     for (int j = 0; j < mask_yellow.cols; j++)
    //     {
    //         mask_yellow.at<uchar>(i, j) = 255 - mask_yellow.at<uchar>(i, j);
    //     }
    // }
    hsv.setTo(Scalar(0, 0, 0), mask_yellow);
    cv::cvtColor(hsv, input, cv::COLOR_HSV2BGR);
    // imshow("no yellow", input);
    // waitKey();

    // Define lower and upper bounds for colors to include
    Scalar lowerb = Scalar(25, 0, 0);
    Scalar upperb = Scalar(255, 255, 255);
    // Create mask to exclude colors outside of bounds
    Mat mask;
    inRange(input, lowerb, upperb, mask);
    Scalar avg_color = mean(input, mask);

    Mat color(100, 100, CV_8UC3, avg_color);
    // imshow("average color", color);
    // waitKey();

    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);
    Scalar hsv_color = hsv.at<cv::Vec3b>(1, 1);

    return hsv_color;
}

Scalar computeAvgColor(Mat img, int id)
{
    if (id != 11)
    {
        return Scalar(0, 0, 0);
    }
    Mat input = img.clone();
    Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    Mat mask;
    inRange(hsv, Scalar(20, 10, 10), Scalar(30, 200, 255), mask);
    for (int i = 0; i < mask.rows; i++)
    {
        for (int j = 0; j < mask.cols; j++)
        {
            mask.at<uchar>(i, j) = 255 - mask.at<uchar>(i, j);
        }
    }
    hsv.setTo(Scalar(0, 0, 0), mask);
    cv::cvtColor(hsv, input, cv::COLOR_HSV2BGR);

    // Define lower and upper bounds for colors to include
    Scalar lowerb = Scalar(25, 0, 0);
    Scalar upperb = Scalar(255, 255, 255);
    // Create mask to exclude colors outside of bounds
    Mat mask2;
    inRange(input, lowerb, upperb, mask2);

    Scalar avg_color = mean(input, mask2);
    Mat color(100, 100, CV_8UC3, avg_color);
    // imshow("average color", color);
    // waitKey();

    cv::cvtColor(color, hsv, cv::COLOR_BGR2HSV);
    Scalar hsv_color = hsv.at<cv::Vec3b>(1, 1);
    cout << "hue pot: " << hsv_color[0] << endl;
    cout << "sat pot: " << hsv_color[1] << endl;
    cout << "vue pot: " << hsv_color[2] << endl;

    return hsv_color;
}

void boundPasta(Mat &dish, Mat &final, std::string label, int id, std::vector<int> forbidden, int max_key, std::vector<FoodData> &foodData)
{
    bool allowed = true;
    cv::Rect box;
    for (int i = 0; i < forbidden.size(); i++)
    {
        if (forbidden[i] == max_key)
        {
            allowed = false;
            break;
        }
    }
    if (allowed)
    {

        box = computeBox(final, dish);
        FoodData bb;
        bb.box = box;
        bb.label = label;
        bb.id = id;
        foodData.push_back(bb);
    }
    return;
}

void boundBread(cv::Mat &input, std::vector<cv::Mat> &dishes,
                cv::Mat &final, std::vector<FoodData> &foodData)
{
    cv::Mat inClone = input.clone();
    cv::Mat grayDish;

    for (cv::Mat &d : dishes)
    {
        cvtColor(d, grayDish, cv::COLOR_BGR2GRAY);
        cv::Mat mask = grayDish > 0;
        inClone.setTo(cv::Scalar(0, 0, 0), mask);
    }

    cv::Mat inHSV;
    cv::cvtColor(inClone, inHSV, cv::COLOR_BGR2HSV);

    Mat mask;
    cv::inRange(inHSV, Scalar(12, 75, 180), Scalar(25, 130, 255), mask);
    for (int i = 0; i < mask.rows; i++)
        for (int j = 0; j < mask.cols; j++)
            mask.at<uchar>(i, j) = 255 - mask.at<uchar>(i, j);

    inHSV.setTo(Scalar(0, 0, 0), mask);

    int kernelSize = 9;
    cv::Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    erode(inHSV, inHSV, kernel);
    kernelSize = 9;
    kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
    dilate(inHSV, inHSV, kernel);

    cv::Mat t;
    cv::cvtColor(inHSV, t, COLOR_HSV2BGR);
    cv::Rect box = computeBox(final, t);
    // imshow("looking for pane", t);
    // waitKey();
    FoodData bb;
    bb.box = box;
    bb.label = "bread";
    bb.id = 13;
    foodData.push_back(bb);
}

void boundSalad(cv::Mat &input, std::vector<cv::Vec3f> accepted_circles,
                cv::Mat &final, std::vector<FoodData> &foodData, std::vector<cv::Mat> &dishes, std::vector<Dish> &dishesData)
{
    cv::Mat inClone;
    cv::Mat grayDish;

    int minRadius = 10000;
    int max_key = -1;
    for (int i = 0; i < accepted_circles.size(); i++)
    {
        cv::Vec3f c = accepted_circles[i];
        if (c[2] < minRadius)
        {
            inClone = 0;
            minRadius = c[2];
            cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
            cv::Point center = cv::Point(c[0], c[1]);
            int radius = c[2];
            cv::circle(mask, center, radius, 255, -1);
            input.copyTo(inClone, mask);
            max_key = i;
        }
    }

    cv::Mat inHSV;
    cv::cvtColor(inClone, inHSV, cv::COLOR_BGR2HSV);

    Mat maskRed;
    cv::inRange(inHSV, Scalar(0, 180, 180), Scalar(20, 255, 255), maskRed);
    Mat maskGreen;
    cv::inRange(inHSV, Scalar(30, 40, 30), Scalar(60, 255, 255), maskGreen);
    for (int i = 0; i < inHSV.rows; i++)
        for (int j = 0; j < inHSV.cols; j++)
            if (maskRed.at<uchar>(i, j) == 0 && maskGreen.at<uchar>(i, j) == 0)
            {
                inHSV.at<cv::Vec3b>(i, j) = 0;
            }
    cv::Mat t;
    cv::cvtColor(inHSV, t, COLOR_HSV2BGR);
    cv::Rect box = computeBox(final, t);

    FoodData bb;
    bb.box = box;
    bb.label = "salad";
    bb.id = 12;
    foodData.push_back(bb);

    Mat segment = Mat::zeros(final.size(), CV_8U);
    for (int i = 0; i < final.rows; i++)
        for (int j = 0; j < final.cols; j++)
            if (dishes[max_key].at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0))
            {
                segment.at<uchar>(i, j) = 255;
            }

    bb.segmentArea = segment;

    dishesData[max_key].addFoods(bb);
}

void boundPotatoes(cv::Mat &dish, cv::Mat &final, std::vector<FoodData> &foodData, std::vector<int> forbidden,
                   int max_key, std::vector<Dish> &dishesData)
{
    bool allowed = true;
    cv::Rect box;
    for (int i = 0; i < forbidden.size(); i++)
    {
        if (forbidden[i] == max_key)
        {
            allowed = false;
            break;
        }
    }
    if (allowed)
    {
        cv::Mat inClone = dish.clone();

        cv::Mat inHSV;
        cv::cvtColor(inClone, inHSV, cv::COLOR_BGR2HSV);

        Mat mask;
        cv::inRange(inHSV, Scalar(20, 10, 100), Scalar(30, 110, 255), mask);
        for (int i = 0; i < mask.rows; i++)
            for (int j = 0; j < mask.cols; j++)
                mask.at<uchar>(i, j) = 255 - mask.at<uchar>(i, j);

        inHSV.setTo(Scalar(0, 0, 0), mask);

        cv::Mat t;
        cv::cvtColor(inHSV, t, COLOR_HSV2BGR);
        // namedWindow("looking for potatoes");
        // imshow("looking for potatoes", t);
        // waitKey();
        cv::Rect box = computeBox(final, t);

        FoodData bb;
        bb.box = box;
        bb.label = "potatoes";
        bb.id = 11;
        foodData.push_back(bb);

        cv::Mat segment = Mat::zeros(t.size(), CV_8U);
        for (int i = 0; i < t.rows; i++)
            for (int j = 0; j < t.cols; j++)
                if (t.at<cv::Vec3b>(i, j) != cv::Vec3b(0, 0, 0))
                {
                    segment.at<uchar>(i, j) = 255;
                }
        int kernelSize = 5;
        Mat kernel = getStructuringElement(MORPH_RECT, Size(kernelSize, kernelSize));
        dilate(inHSV, inHSV, kernel);
        bb.segmentArea = segment;
        dishesData[max_key].addFoods(bb);
    }
}

void drawBoundingBoxes(cv::Mat &final, std::vector<FoodData> &foodData)
{
    for (FoodData &bb : foodData)
    {
        if (bb.box.y - 20 > 0)
        {
            cv::putText(final, bb.label, cv::Point(bb.box.x, bb.box.y - 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 0.8, 8, false);
        }
        else
        {
            cv::putText(final, bb.label, cv::Point(bb.box.x, bb.box.y + bb.box.height + 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 0.8, 8, false);
        }
        cv::rectangle(final, bb.box, CV_RGB(255, 0, 0), 2);
    }
}

cv::Rect computeBox(cv::Mat &final, cv::Mat &dish)
{
    int x = final.cols;
    int y = final.rows;
    int max_x = 0;
    int max_y = 0;
    for (int i = 0; i < dish.cols; i++)
    {
        for (int j = 0; j < dish.rows; j++)
        {
            if (dish.at<cv::Vec3b>(cv::Point(i, j))[0] != 0 &&
                dish.at<cv::Vec3b>(cv::Point(i, j))[1] != 0 &&
                dish.at<cv::Vec3b>(cv::Point(i, j))[2] != 0)
            {
                if (i < x)
                {
                    x = i;
                }
                if (i > max_x)
                {
                    max_x = i;
                }
                if (j < y)
                {
                    y = j;
                }
                if (j > max_y)
                {
                    max_y = j;
                }
            }
        }
    }
    return cv::Rect(x, y, max_x - x, max_y - y);
}
