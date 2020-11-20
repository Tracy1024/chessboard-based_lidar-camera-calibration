cmake_minimum_required(VERSION 3.10)
project(demo_projection)

find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(demo_projection main.cpp)
target_link_libraries (demo_projection ${OpenCV_LIBS} ${PCL_LIBRARIES} Eigen3::Eigen)