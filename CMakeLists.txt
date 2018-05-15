cmake_minimum_required(VERSION 2.8)
project(practice)
find_package(OpenCV REQUIRED)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_executable( practice video_test.cpp )
target_compile_features(practice PUBLIC cxx_lambdas)
target_link_libraries( practice ${OpenCV_LIBS} )

