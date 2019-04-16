cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
PROJECT ("mrcncpp")
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
 
# 빌드 형상(Configuration) 및 주절주절 Makefile 생성 여부
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH "/root/libtorch")
set(OUTPUT
    "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.out")
set(SRC_FILES
    example-app.cpp
    include/layers/batch_norm.h
    include/layers/ROIAlign.h
    include/utils/nms.h
    source/layers/ROIAlign.cpp
    source/utils/nms.cpp
    source/layers/misc.cpp
    source/layers/smooth_l1_loss.cpp
    include/layers/misc.h
    include/layers/smooth_l1_loss.h
    )

find_package(Torch REQUIRED)

add_executable(${OUTPUT} ${SRC_FILES})
target_link_libraries(${OUTPUT} "${TORCH_LIBRARIES}")
set_property(TARGET ${OUTPUT} PROPERTY CXX_STANDARD 11)

target_include_directories(${OUTPUT} PUBLIC include)