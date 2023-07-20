#ifndef DISH_H
#define DISH_H

#include "utils.h"
#include <vector>

class Dish
{
private:
    std::vector<FoodData> foods;
    cv::Mat dish;

public:
    std::vector<FoodData> getFoods() const { return foods; }
    cv::Mat getDish() const { return dish; }
    void setFoods(const std::vector<FoodData> &newFoods) { foods = newFoods; }
    void addFoods(const FoodData &newFood) { foods.push_back(newFood); }
    void setDish(const cv::Mat &newDish) { dish = newDish; }
};

#endif