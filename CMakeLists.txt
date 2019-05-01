cmake_minimum_required(VERSION 3.8.2 FATAL_ERROR)


PROJECT ("rcnncpp")
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
 
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH "/root/maskrcnn/lib/libtorch")
set(OUTPUT
    "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.out")
if(DEFINED ENV{CUDA_HOME})
  enable_language(CUDA)
  set(CUDA_NVCC_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__")
endif()


set(SRC_FILES
    example-app.cpp
    )

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/yaml-cpp)
find_package(yaml-cpp REQUIRED)
find_package(Torch REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/yaml-cpp/include)

add_executable(${OUTPUT} ${SRC_FILES})
target_link_libraries(${OUTPUT} "${TORCH_LIBRARIES}" structures config yaml-cpp layers)
set_property(TARGET ${OUTPUT} PROPERTY CXX_STANDARD 11)