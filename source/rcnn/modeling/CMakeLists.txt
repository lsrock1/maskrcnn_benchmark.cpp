add_library(modeling backbone/resnet.cpp)
target_include_directories(modeling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/modeling/backbone/)
target_link_libraries(modeling "${TORCH_LIBRARIES}")