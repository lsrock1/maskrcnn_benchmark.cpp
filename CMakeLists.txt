cmake_minimum_required(VERSION 3.12.3 FATAL_ERROR)

PROJECT ("rcnncpp")
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
 
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)
set(OUTPUT  "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.out")
find_package(CUDA)

set(SRC_FILES
    example-app.cpp)

if(CUDA_FOUND)
  message("CUDA FOUND")
  enable_language(CUDA)
  string(APPEND CMAKE_CUDA_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__")
endif()
find_package(Torch REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source)
add_executable(${OUTPUT} ${SRC_FILES})
target_link_libraries(${OUTPUT} ${TORCH_LIBRARIES} modeling config)
set_property(TARGET ${OUTPUT} PROPERTY CXX_STANDARD 11)