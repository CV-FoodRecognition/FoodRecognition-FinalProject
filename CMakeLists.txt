cmake_minimum_required(VERSION 3.1)
project(FinalProject)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(headers)

add_executable(main main.cpp headers/descriptor_methods.h
  headers/matcher_methods.h
  functions/detect_recognition.cpp
  functions/Leftover.cpp
  functions/utils.cpp 
  functions/segmentation.cpp
  functions/descriptor_methods.cpp
  functions/ImageProcessor.cpp
  functions/matcher_methods.cpp)
target_link_libraries(main ${OpenCV_LIBS})

## to run:  
#
#   ./main 
#