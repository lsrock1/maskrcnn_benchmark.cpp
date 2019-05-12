file(GLOB sources "backbone/*.cpp" "rpn/*.cpp" "*.cpp")
add_library(modeling ${sources})
target_include_directories(modeling  
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/modeling/backbone/
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/modeling/rpn/
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/modeling/ )
target_link_libraries(modeling layers config)