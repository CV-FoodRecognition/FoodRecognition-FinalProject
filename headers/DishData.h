#ifndef DISHDATA_CLASS_H
#define DISHDATA_CLASS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"

class DishData
{
private:
    cv::Mat dish;                 // associated dish
    std::vector<int> ids;         // all the ids of the food in the dish
    std::vector<BoundingBox> bbs; // bounding boxes
    std::vector<Area> areas;      // structure for the areas of the segments

public:
    // Getters
    cv::Mat getDish() const { return dish; }
    std::vector<int> getIds() const { return ids; }
    std::vector<BoundingBox> getBoundingBoxes() const { return bbs; }
    std::vector<Area> getAreas() const { return areas; }
};

#endif