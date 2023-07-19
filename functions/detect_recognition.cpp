#include "detect_recognition.h"

using namespace cv;
using namespace std;

/*
Class written by @rickyvendra
*/

void detectAndCompute(cv::Mat in1, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches,
                      std::vector<cv::Vec3f> accepted_circles, std::vector<FoodData> &foodData, std::vector<foodTemplate> templates,
                      cv::Mat &final)
{
    Result result;
    std::vector<int> forbidden;

    bool alreadyFoundFirstDish = false;
    for (int f = 0; f < templates.size(); f++)
    { // for every food
        cout << "STO CERCANDO: " << templates[f].label << endl;
        for (int d = 0; d < dishes.size(); d++)
        {
            dishesMatches[d] = 0;
        }
        cout << "inizio a cercare match" << endl;
        for (int t = 0; t < templates[f].foodTemplates.size(); t++)
        { // for every food template
            for (int d = 0; d < dishes.size(); d++)
            { // look for matches in every dish
                result = useDescriptor(dishes[d], templates[f].foodTemplates[t], DescriptorType::SIFT);
                dishesMatches[d] = dishesMatches[d] + bruteForceKNN(dishes[d], templates[f].foodTemplates[t], result);
                std::cout << "matches dishes " << d << " : " << dishesMatches[d] << std::endl;
            }
        }
        cout << "finito di cercare i match" << endl;
        int max_key = 0;
        Scalar avgColor;
        if (templates[f].id != 13)
        {
            max_key = computeBestDish(templates[f], dishes, dishesMatches);
            if (max_key > -1)
            {
                avgColor = computeAvgColor(dishes[max_key]);
            }
        }

        // compute best dish
        // max_key = computeBestDish(templates[f], dishes, dishesMatches);
        cout << "max key" << max_key << endl;

        cout << "trovato il piatto con pi� match" << endl;

        cout << "match finale (quello pi� giusto)" << endl;
        if (max_key > -1)
        {
            if (templates[f].id == 1 && avgColor[0] > 30 && avgColor[0] < 50 && !alreadyFoundFirstDish)
            {
                // pesto
                boundPasta(dishes[max_key], final, templates[f].label, templates[f].id, forbidden, max_key, foodData);
                forbidden.push_back(max_key);
                alreadyFoundFirstDish = true;
            }
            else if ((templates[f].id == 2 || templates[f].id == 3) && avgColor[0] > 0 && avgColor[0] < 11 && avgColor[1] > 130 && !alreadyFoundFirstDish)
            {
                // tomato
                std::vector<std::string> labels;
                labels.push_back("pasta with tomato sauce");
                labels.push_back("pasta with meat sauce");
                std::vector<int> ids;
                ids.push_back(2);
                ids.push_back(3);
                boundPasta(dishes[max_key], final, labels, ids, forbidden, max_key, foodData);
                forbidden.push_back(max_key);
                alreadyFoundFirstDish = true;
            }
            else if (templates[f].id == 4 && avgColor[0] > 10 && avgColor[0] < 20 && avgColor[1] > 140 && !alreadyFoundFirstDish)
            {
                // pesto
                boundPasta(dishes[max_key], final, templates[f].label, templates[f].id, forbidden, max_key, foodData);
                forbidden.push_back(max_key);
                alreadyFoundFirstDish = true;
            }
            else if (templates[f].id == 13)
            {
                // bread
                boundBread(in1, dishes, final, foodData);
            }
            else if (templates[f].id == 11)
            {
                // potatoes
                boundPotatoes(dishes[max_key], final, foodData, forbidden, max_key);
            }
            else if (dishes.size() > 2 && templates[f].id == 12)
            {
                // salad
                boundSalad(in1, accepted_circles, final, foodData);
            }
            else if (templates[f].id > 4)
            {
                bool allowed = true;
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
                    bruteForceKNN(in1, templates[f], dishes[max_key], final, foodData);
                    if (templates[f].id == 5)
                    {
                        forbidden.push_back(max_key);
                    }
                }
            }
        }
    }
    drawBoundingBoxes(final, foodData);
}

int computeBestDish(foodTemplate food, std::vector<cv::Mat> dishes, std::vector<int> dishesMatches)
{
    int best_dish_id = -1;
    for (int d = 0; d < dishesMatches.size(); d++)
    {
        Scalar avgColor = computeAvgColor(dishes[d]);
        if (food.id == 1)
        {
            if (avgColor[0] > 30 && avgColor[0] < 50)
            {
                return d;
            }
        }
        else if (food.id == 2 || food.id == 3)
        {
            if (avgColor[0] > 0 && avgColor[0] < 11 && avgColor[1] > 130)
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
        else if (dishesMatches[d] > dishesMatches[best_dish_id])
        {
            best_dish_id = d;
        }
    }
    return best_dish_id;
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
        bb.labels.push_back(label);
        bb.ids.push_back(id);
        foodData.push_back(bb);
    }
    return;
}

void boundPasta(Mat &dish, Mat &final, std::vector<std::string> labels, std::vector<int> ids, std::vector<int> forbidden,
                int max_key, std::vector<FoodData> &foodData)
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
        bb.src = dish;
        bb.box = box;
        for (std::string label : labels)
        {
            bb.labels.push_back(label);
        }
        for (int id : ids)
        {
            bb.ids.push_back(id);
        }
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
    bb.src = input;
    bb.labels.push_back("bread");
    bb.ids.push_back(13);
    foodData.push_back(bb);
}

void boundSalad(cv::Mat &input, std::vector<cv::Vec3f> accepted_circles,
                cv::Mat &final, std::vector<FoodData> &foodData)
{
    cv::Mat inClone;
    cv::Mat grayDish;

    int minRadius = 10000;
    for (cv::Vec3f &c : accepted_circles)
    {
        if (c[2] < minRadius)
        {
            inClone = 0;
            minRadius = c[2];
            cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);
            cv::Point center = cv::Point(c[0], c[1]);
            int radius = c[2];
            cv::circle(mask, center, radius, 255, -1);
            input.copyTo(inClone, mask);
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
    // namedWindow("looking for salad");
    // imshow("looking for salad", t);
    // waitKey();
    cv::Rect box = computeBox(final, t);

    FoodData bb;
    bb.src = input;
    bb.box = box;
    bb.labels.push_back("salad");
    bb.ids.push_back(12);
    foodData.push_back(bb);
}

void boundPotatoes(cv::Mat &dish, cv::Mat &final, std::vector<FoodData> &foodData, std::vector<int> forbidden, int max_key)
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
        cv::Mat grayDish;

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
        bb.src = dish;
        bb.box = box;
        bb.labels.push_back("potatoes");
        bb.ids.push_back(11);
        foodData.push_back(bb);
    }
}

void drawBoundingBoxes(cv::Mat &final, std::vector<FoodData> &foodData)
{
    for (FoodData &bb : foodData)
    {
        std::string text;
        int numberFoods = bb.labels.size();
        for (std::string label : bb.labels)
        {
            text = text + std::to_string(100 / numberFoods) + "% " + label + " ";
        }
        if (bb.box.y - 20 > 0)
        {
            cv::putText(final, text, cv::Point(bb.box.x, bb.box.y - 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 0.8, 8, false);
        }
        else
        {
            cv::putText(final, text, cv::Point(bb.box.x, bb.box.y + bb.box.height + 20), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 0.8, 8, false);
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

Scalar computeAvgColor(Mat img)
{
    Mat input = img.clone();
    // imshow("dish", input);
    // waitKey();
    Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);

    Mat mask_yellow;
    inRange(hsv, Scalar(14, 0, 0), Scalar(25, 255, 255), mask_yellow);
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
    cout << "hue: " << hsv_color[0] << endl;
    cout << "sat: " << hsv_color[1] << endl;
    cout << "vue: " << hsv_color[2] << endl;

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