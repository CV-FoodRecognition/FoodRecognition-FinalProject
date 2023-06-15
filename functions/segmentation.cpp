#include "../headers/segmentation.h"
#include "../headers/utils.h"

using namespace cv;
using namespace std;

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
    imshow("New Sharped Image", imgResult);
    waitKey();

    return imgResult;

    // Create binary image from source image
    // Mat bw;
    // cvtColor(imgResult, bw, COLOR_BGR2GRAY);
    // threshold(bw, bw, 220, 255, THRESH_OTSU);
    // imshow("thershold segmentation", bw);
}

void kmeansSegmentation(int k, cv::Mat &src)
{
    cv::Mat shifted;
    cv::pyrMeanShiftFiltering(src, shifted, 80, 150);

    std::vector<int> labels;
    cv::Mat1f colors;
    int attempts = 10;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 20, 1.);

    cv::Mat input = shifted.reshape(1, src.rows * src.cols);
    input.convertTo(input, CV_32F);

    cv::kmeans(input, k, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, colors);

    std::vector<cv::Mat> masks(k);
    for (int i = 0; i < k; i++)
        masks[i] = cv::Mat::zeros(src.size(), CV_8U);

    for (int i = 0; i < src.rows * src.cols; i++)
        masks[labels[i]].at<uchar>(i / src.cols, i % src.cols) = 255;

    std::vector<cv::Mat> results(k);
    for (int i = 0; i < k; i++)
        src.copyTo(results[i], masks[i]);

    for (int i = 0; i < k; i++)
    {
        std::string windowName = "mask" + std::to_string(i + 1);
        showImg(windowName, results[i]);
    }
    cv::waitKey();
}
