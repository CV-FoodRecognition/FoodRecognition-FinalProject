#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>

/*
 *   Strategy pattern to have easier code for preprocessing
 */

class ImagePreprocessor
{
public:
    using PreprocessingFunction = std::function<cv::Mat(cv::Mat)>;

    void addPreprocessingFunction(PreprocessingFunction function)
    {
        preprocessingFunctions.push_back(function);
    }

    cv::Mat preprocessImage(cv::Mat image)
    {
        for (const auto &function : preprocessingFunctions)
        {
            image = function(image);
        }
        return image;
    }

private:
    std::vector<PreprocessingFunction> preprocessingFunctions;
};

// ImagePreprocessor preprocessor;
// preprocessor.addPreprocessingFunction(sharpenImg);

// cv::Mat preprocessedImage = preprocessor.preprocessImage(in1);
// cv::Mat preprocessedImage = preprocessor.preprocessImage(in2);
