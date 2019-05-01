add_library(config defaults.cpp)
target_include_directories(config PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/config/)
target_link_libraries(config yaml-cpp)