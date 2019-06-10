add_library(cocotool mask_api.cpp coco.cpp)

target_include_directories(cocotool 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include/coco/ )
# target_link_libraries(cococotool)
