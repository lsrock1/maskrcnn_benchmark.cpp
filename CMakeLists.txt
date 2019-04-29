cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
PROJECT ("rcnncpp")
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 1)
 
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_VERBOSE_MAKEFILE true)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_PREFIX_PATH "/root/maskrcnn/lib/libtorch")
set(OUTPUT
    "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.out")


set(SRC_FILES
    example-app.cpp
    )

find_package(Torch REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/source)



add_executable(${OUTPUT} ${SRC_FILES})
target_link_libraries(${OUTPUT} "${TORCH_LIBRARIES}" structures)
set_property(TARGET ${OUTPUT} PROPERTY CXX_STANDARD 11)