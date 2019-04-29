add_library(structures bounding_box.cpp)
target_include_directories(structures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/structures/)
target_link_libraries(structures "${TORCH_LIBRARIES}")