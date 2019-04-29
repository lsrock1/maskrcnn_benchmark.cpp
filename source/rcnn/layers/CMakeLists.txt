add_library(layers batch_norm.cpp
    misc.cpp
    ROIAlign.cpp
    smooth_l1_loss.cpp)
target_include_directories(layers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/)
target_link_libraries(layers "${TORCH_LIBRARIES}")