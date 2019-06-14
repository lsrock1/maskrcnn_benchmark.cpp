add_library(data coco_detection.cpp)
target_include_directories(data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/data/ ${OpenCV_INCLUDE_DIRS})
target_link_libraries(data PUBLIC ${TORCH_LIBRARIES} structures cocotool PRIVATE ${OpenCV_LIBS})