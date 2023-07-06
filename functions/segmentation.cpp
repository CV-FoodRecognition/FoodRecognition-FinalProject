#include "../headers/segmentation.h"

using namespace cv;
using namespace std;

extern std::string window_name;
extern int low_k;
extern cv::Mat src;
extern std::vector<int> kmeans_labels;
extern cv::Mat1f colors;

void doHough(std::vector<cv::Mat> &dishes, std::vector<int> &dishesMatches, Mat &in)
{

    cv::Mat in_gray;
    cvtColor(in, in_gray, cv::COLOR_BGR2GRAY);
    in_gray.convertTo(in_gray, CV_8UC1);
    cv::GaussianBlur(in_gray, in_gray, cv::Size(7, 7), 1.5, 1.5, 4);
    // Hough Circles per ottenere solo i piatti
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(in_gray, circles, cv::HOUGH_GRADIENT,
                     1, 1, 100, 85,
                     150, 500); // min radius and max radius

    std::vector<cv::Vec3f> accepted_circles;

    for (int k = 0; k < circles.size(); k++)
    {
        cv::Vec3i c = circles[k];
        cv::Point center = cv::Point(c[0], c[1]);
        int radius = c[2];

        cv::Mat mask = cv::Mat::zeros(in.size(), CV_8UC1);
        cv::Mat dish = cv::Mat::zeros(in.size(), CV_8UC3);

        if (accepted_circles.size() == 0)
        {
            dishesMatches.push_back(0);
            cv::circle(mask, center, radius, 255, -1);
            Mat mask_colored;
            in.copyTo(mask_colored, mask);
            dishes.push_back(mask_colored);
            accepted_circles.push_back(c);
        }
        if (!isInside(accepted_circles, center))
        {
            dishesMatches.push_back(0);
            cv::circle(mask, center, radius, 255, -1);
            Mat mask_colored;
            in.copyTo(mask_colored, mask);
            dishes.push_back(mask_colored);
            accepted_circles.push_back(c);
        }
    }
    for (int i = 0; i < accepted_circles.size(); i++)
    {
        cv::Vec3i c = accepted_circles[i];
        cout << "centro esistente: " << c[0] << " , " << c[1] << endl;
        cout << "raggio:  " << c[2] << endl;
    }
}

bool isInside(std::vector<cv::Vec3f> circles, cv::Point center)
{
    for (int i = 0; i < circles.size(); i++)
    {
        cv::Vec3i c = circles[i];
        cv::Point existingCenter = cv::Point(c[0], c[1]);
        cout << "centro esistente: " << c[0] << " , " << c[1] << endl;
        cout << "centro nuovo: " << center.x << " , " << center.y << endl;
        double distance = cv::norm(existingCenter - center);
        cout << "distanza: " << distance << "   raggio:  " << c[2] << endl;
        if (distance < c[2])
        {
            cout << "scartato" << endl;
            return true;
        }
    }
    cout << "va bene" << endl;
    return false;
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
    // Pyramidal Filtering with Mean Shift to have a CARTOONISH effect on input image
    // cv::pyrMeanShiftFiltering(src, src, 18, 150);
    // showImg("Cartoonish (MeanShift Filter)", shifted);

    std::vector<int> labels;
    cv::Mat1f colors;
    int attempts = 5;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.);

    cv::Mat input = src.reshape(1, src.rows * src.cols);
    input.convertTo(input, CV_32F);

    cv::kmeans(input, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, colors);

    vector<pair<int, int>> clusterCounts(k);
    for (int i = 0; i < labels.size(); i++)
    {
        clusterCounts[labels[i]].first++;            // count num of pixels per cluster
        clusterCounts[labels[i]].second = labels[i]; // index
    }

    // clusters by number of pixels
    sort(clusterCounts.rbegin(), clusterCounts.rend());

    // cluster colors
    vector<Vec3b> clusterColors(k);
    clusterColors[0] = Vec3b(0, 0, 0);     // black -- background
    clusterColors[1] = Vec3b(0, 255, 255); // yellow  -- first food per pixel
    clusterColors[2] = Vec3b(255, 0, 0);   // blue  -- second food per pixel
    clusterColors[3] = Vec3b(0, 255, 0);   // green -- additional color
    clusterColors[4] = Vec3b(0, 0, 255);   // red   -- additional color

    // assign colors to the clusters (descending order for num of pixels assigned in each cluster)
    vector<Vec3b> sortedClusterColors(k);
    for (int i = 0; i < k; i++)
        sortedClusterColors[clusterCounts[i].second] = clusterColors[i];

    // create output image
    Mat output(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows * src.cols; i++)
        output.at<Vec3b>(i / (src.cols), i % (src.cols)) = sortedClusterColors[labels[i]];

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
