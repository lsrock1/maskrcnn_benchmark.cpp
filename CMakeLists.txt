cmake_minimum_required(VERSION 3.12.3 FATAL_ERROR)

PROJECT ("MRCNN_EXAMPLE")

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_STANDARD 11)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)
if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
  message("ABI SET 0")
  set(MRCN_ABI_FLAG "CMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

find_package(Torch REQUIRED)
find_package(CUDA)
find_package(OpenCV REQUIRED)

# if(NOT OpenCV_FOUND)
#   execute_process (
#     COMMAND bash -c
#     COMMAND bash -c "./lib/opencv"
#     OUTPUT_VARIABLE outVar
#   )
# endif()

if(CUDA_FOUND)
  message("CUDA FOUND")
  enable_language(CUDA)
  set(CUDA_STANDARD 11)
  set(CUDA_STANDARD_REQUIRED ON)
  string(APPEND CMAKE_CUDA_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -std=c++11")
endif()

# add_subdirectory(lib/yaml-cpp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/gtest)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/rcnn)
add_executable(tmp.out
  tmp_test.cpp)

target_link_libraries(tmp.out maskrcnn cocotool)
