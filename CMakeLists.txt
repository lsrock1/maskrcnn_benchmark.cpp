cmake_minimum_required(VERSION 3.12.3 FATAL_ERROR)

PROJECT ("MRCNN_EXAMPLE")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)

find_package(Torch)
find_package(CUDA)
find_package(OpenCV)
find_package(yaml-cpp)

# Torch Not found

if(CUDA_FOUND)
  message("CUDA FOUND")
  enable_language(CUDA)
  set(CUDA_STANDARD 11)
  set(CUDA_STANDARD_REQUIRED ON)
  string(APPEND CMAKE_CUDA_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -std=c++11")
endif()

if(NOT OpenCV_FOUND)
  message("OpenCV is Not Found")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv)
endif()

if(NOT yaml-cpp_FOUND)
  message("yaml-cpp is not found")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/yaml-cpp)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/gtest)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/rcnn)
add_executable(run.out runner.cpp)

target_link_libraries(run.out maskrcnn cocotool)
