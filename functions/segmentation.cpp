#include "../headers/segmentation.h"

using namespace cv;
using namespace std;

extern std::string window_name;
extern int low_k;
extern cv::Mat src;
extern std::vector<int> kmeans_labels;
extern cv::Mat1f colors;

void doHough(std::vector<cv::Mat> &dishes, Mat &in, Mat &in_gray)
{
    // Blur to remove possible noise
    cv::GaussianBlur(in_gray, in_gray, cv::Size(7, 7), 1.5, 1.5, 4);

    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(in_gray, circles, cv::HOUGH_GRADIENT,
                     1, 40, 100, 100,
                     150, 400); // min radius & max radius

    for (int k = 0; k < circles.size(); k++)
    {
        cv::Mat mask = cv::Mat::zeros(in.size(), CV_8UC1);
        cv::Mat dish = cv::Mat::zeros(in.size(), CV_8UC3);
        dishes.push_back(dish);
        cv::Vec3i c = circles[k];
        cv::Point center = cv::Point(c[0], c[1]); // c0 = x coord , c1 = y coord of the circle
        int radius = c[2];                        // c2 = ray of the circle
        circle(mask, center, radius, 255, -1);
        in.copyTo(dishes[k], mask);

        // showImg("dishes", dishes[k]);
    }
}

void doMSER(std::vector<cv::Rect> &mser_bbox, cv::Mat shifted, Mat result)
{
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    std::vector<std::vector<cv::Point>> regions;
    ms->detectRegions(shifted, regions, mser_bbox);

    for (int i = 0; i < regions.size(); i++)
    {
        cv::rectangle(shifted, mser_bbox[i], CV_RGB(0, 255, 0));

        cv::Mat mask, bg, fg;

        // grabCut(shifted, mask, mser_bbox[i], bg, fg, 1, GC_INIT_WITH_RECT);
        Rect rect = mser_bbox[i];
        int area = rect.width * rect.height;
        for (int i = rect.x; i < rect.x + rect.width; i++)
        {
            for (int j = rect.y; j < rect.y + rect.height; j++)
            {
                result.at<Vec3b>(Point(i, j))[0] = 0;
                result.at<Vec3b>(Point(i, j))[1] = 0;
                result.at<Vec3b>(Point(i, j))[2] = 255;
            }
        }
    }
}

Scalar computeAvgColor(Mat shifted, Rect box)
{
    Mat roi = shifted(box);

    // Define lower and upper bounds for colors to include
    Scalar lowerb = Scalar(15, 15, 15);
    Scalar upperb = Scalar(240, 240, 240);

    // Create mask to exclude colors outside of bounds
    Mat mask;
    inRange(roi, lowerb, upperb, mask);
    Scalar avg_color = mean(roi, mask);
    Mat image(100, 100, CV_8UC3, avg_color);
    // showImg("average color", image);

    return avg_color;
}

Mat meanShiftFunct(Mat src)
{
    cv::Mat shifted;
    cv::pyrMeanShiftFiltering(src, shifted, 15, 45);
    imshow("Mean Shifted", shifted);

    // Change the background from white to black, since that will help later to extract
    // better results during the use of Distance Transform
    Mat mask;
    inRange(shifted, Scalar(80, 80, 80), Scalar(255, 255, 255), mask);
    src.setTo(Scalar(255, 255, 255), mask);
    // Show output image
    // imshow("Black Background Image", src);

    // Create a kernel that we will use to sharpen our image
    Mat kernel = (Mat_<float>(3, 3) << 1, 1, 1,
                  1, -8, 1,
                  1, 1, 1); // an approximation of second derivative, a quite strong kernel
    // do the laplacian filtering as it is
    // well, we need to convert everything in something more deeper then CV_8U
    // because the kernel has some negative values,
    // and we can expect in general to have a Laplacian image with negative values
    // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    // so the possible negative number will be truncated
    Mat imgLaplacian;
    filter2D(src, imgLaplacian, CV_32F, kernel);
    Mat sharp;
    src.convertTo(sharp, CV_32F);
    Mat imgResult = sharp - imgLaplacian;
    // convert back to 8bits gray scale
    imgResult.convertTo(imgResult, CV_8UC3);
    imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
    // imshow( "Laplace Filtered Image", imgLaplacian );
    // imshow("New Sharped Image", imgResult);
    waitKey();

    return imgResult;

    // Create binary image from source image
    // Mat bw;
    // cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    // threshold(bw, bw, 220, 255, THRESH_OTSU);
    // imshow("thershold segmentation", bw);
}

void equalizeHistogram(cv::Mat &src, cv::Mat &dst)
{
    // Convert the image to the lab color space
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // Split the channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);

    // Equalize the histogram of the Y channel
    cv::equalizeHist(labChannels[0], labChannels[0]);

    // Merge the channels back and convert back to BGR color space
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}

void removeBackground(cv::Mat &src)
{
    for (int k = 255; k > 20; k = k - 5)
    {
        Mat mask;
        inRange(src, Scalar(k - 40, k - 40, k - 40), Scalar(k, k, k), mask);
        src.setTo(Scalar(0, 0, 0), mask);
    }
}

cv::Mat kmeansSegmentation(int k, cv::Mat &src)
{
    std::vector<int> labels;
    cv::Mat1f colors;
    int attempts = 5;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.);

    cv::Mat input = src.reshape(1, src.rows * src.cols);
    input.convertTo(input, CV_32F);

    cv::kmeans(input, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, colors);

    // creats output image
    cv::Mat output(src.size(), CV_8UC3);

    // defines colors for each cluster
    std::vector<cv::Vec3b> cluster_colors(k);
    for (int i = 0; i < k; i++)
        cluster_colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);

    // sets pixel values in output image based on cluster assignments
    for (int i = 0; i < src.rows * src.cols; i++)
        output.at<cv::Vec3b>(i / src.cols, i % src.cols) = cluster_colors[labels[i]];

    return output;
}

void removeShadows(cv::Mat &src, cv::Mat &dst)
{
    // Converting image to LAB color space
    cv::Mat lab;
    cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

    // Splitting the channels
    std::vector<cv::Mat> labChannels(3);
    cv::split(lab, labChannels);

    // Apply Contrast Limited Adaptive Histogram Equalization
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(1);
    cv::Mat enhancedL;
    clahe->apply(labChannels[0], enhancedL);

    // Merge the channels back and convert back to BGR color space
    enhancedL.copyTo(labChannels[0]);
    cv::merge(labChannels, lab);
    cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
}
