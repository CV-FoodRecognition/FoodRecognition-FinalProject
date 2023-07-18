#include "../headers/ImageProcessor.h"

void ImageProcessor::doHough(cv::Mat &in)
{
    cv::Mat in_gray, temp;
    cvtColor(in, in_gray, cv::COLOR_BGR2GRAY);
    in_gray.convertTo(in_gray, CV_8UC1);
    cv::GaussianBlur(in_gray, in_gray, cv::Size(3, 3), 0.5);

    temp = in.clone();

    // Hough Circles per ottenere solo i piatti
    /*std::vector<cv::Vec3f> circles;
    cv::HoughCircles(in_gray, circles, cv::HOUGH_GRADIENT,
                     1, in_gray.rows / 2.5, 140, 55,
                     185, 370); // min radius and max radius
*/

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
        radia.push_back(radius);

        cv::Mat mask = cv::Mat::zeros(in.size(), CV_8UC1);
        cv::Mat dish = cv::Mat::zeros(in.size(), CV_8UC3);

        acceptCircles(in, mask, temp,
                      c, center, radius,
                      accepted_circles,
                      dishesMatches, dishes);
    }
    for (int i = 0; i < accepted_circles.size(); i++)
        cv::Vec3i c = accepted_circles[i];
}

void ImageProcessor::doMSER(cv::Mat &shifted, cv::Mat &result)
{
    std::cout << "1 start" << std::endl;
    cv::Mat cloneShifted = shifted.clone();
    cv::Ptr<cv::MSER> ms = cv::MSER::create();
    ms->detectRegions(cloneShifted, regions, mser_bbox);
    std::cout << "2 detection" << std::endl;

    for (int i = 0; i < regions.size(); i++)
    {
        cv::rectangle(cloneShifted, mser_bbox[i], CV_RGB(0, 255, 0));
        cv::Mat mask, bg, fg;

        std::cout << "3" << std::endl;

        cv::Rect rect = mser_bbox[i];
        int area = rect.width * rect.height;
        for (int i = rect.x; i < rect.x + rect.width; i++)
        {
            for (int j = rect.y; j < rect.y + rect.height; j++)
            {
                result.at<cv::Vec3b>(cv::Point(i, j))[0] = 0;
                result.at<cv::Vec3b>(cv::Point(i, j))[1] = 0;
                result.at<cv::Vec3b>(cv::Point(i, j))[2] = 255;
            }
            std::cout << "4" << std::endl;
        }
        std::cout << "5 fine" << std::endl;
    }
}

cv::Mat ImageProcessor::kmeansSegmentation(int k, cv::Mat &src)
{
    std::vector<int> labels;
    cv::Mat1f colors;
    int attempts = 5;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.);

    cv::Mat input = src.reshape(1, src.rows * src.cols);
    input.convertTo(input, CV_32F);

    cv::kmeans(input, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, colors);

    std::vector<std::pair<int, int>> clusterCounts(k);
    for (int i = 0; i < labels.size(); i++)
    {
        clusterCounts[labels[i]].first++;            // count num of pixels per cluster
        clusterCounts[labels[i]].second = labels[i]; // index
    }

    // clusters by number of pixels
    sort(clusterCounts.rbegin(), clusterCounts.rend());

    // cluster colors
    std::vector<cv::Vec3b> clusterColors(k);
    clusterColors[0] = cv::Vec3b(0, 0, 0);     // black -- background
    clusterColors[1] = cv::Vec3b(0, 255, 255); // yellow-- first food per pixel
    clusterColors[2] = cv::Vec3b(255, 0, 0);   // blue  -- second food per pixel
    clusterColors[3] = cv::Vec3b(0, 255, 0);   // green -- additional color
    clusterColors[4] = cv::Vec3b(0, 0, 255);   // red   -- additional color

    // assign colors to the clusters (descending order for num of pixels assigned in each cluster)
    std::vector<cv::Vec3b> sortedClusterColors(k);
    for (int i = 0; i < k; i++)
        sortedClusterColors[clusterCounts[i].second] = clusterColors[i];

    // create output image
    cv::Mat output(src.size(), CV_8UC3);
    for (int i = 0; i < src.rows * src.cols; i++)
        output.at<cv::Vec3b>(i / (src.cols), i % (src.cols)) = sortedClusterColors[labels[i]];

    return output;
}

// Helper function for Hough
bool isInside(std::vector<cv::Vec3f> circles, cv::Point center)
{
    for (int i = 0; i < circles.size(); i++)
    {
        cv::Vec3i c = circles[i];
        cv::Point existingCenter = cv::Point(c[0], c[1]);
        // std::cout << "centro esistente: " << c[0] << " , " << c[1] << std::endl;
        // std::cout << "centro nuovo: " << center.x << " , " << center.y << std::endl;
        double distance = cv::norm(existingCenter - center);
        // std::cout << "distanza: " << distance << "   raggio:  " << c[2] << std::endl;
        if (distance < c[2])
        {
            // std::cout << "scartato" << std::endl;
            return true;
        }
    }
    // std::cout << "va bene" << std::endl;
    return false;
}