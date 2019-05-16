cmake_minimum_required(VERSION 3.12.3 FATAL_ERROR)

PROJECT ("MRCNN_EXAMPLE")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)

find_package(Torch REQUIRED)
find_package(CUDA)

if(CUDA_FOUND)
  message("CUDA FOUND")
  enable_language(CUDA)
  string(APPEND CMAKE_CUDA_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/gtest/googletest/include)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source)
#can't test, issue https://github.com/pytorch/pytorch/issues/19353
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/gtest)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/rcnn)
add_executable(tmp.out
  tmp_test.cpp)

target_link_libraries(tmp.out maskrcnn)