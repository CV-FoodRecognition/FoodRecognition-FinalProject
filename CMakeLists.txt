cmake_minimum_required(VERSION 3.1)
project(FinalProject)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp headers/descriptor_methods headers/utils headers/segmentation)
target_link_libraries(main ${OpenCV_LIBS})

## to run:  
#
#   ./main 
#