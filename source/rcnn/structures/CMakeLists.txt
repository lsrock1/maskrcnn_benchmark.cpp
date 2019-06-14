add_library(structures bounding_box.cpp image_list.cpp segmentation_mask.cpp)
target_include_directories(structures 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/structures/)
target_link_libraries(structures PUBLIC ${TORCH_LIBRARIES} PRIVATE layers cocotool)