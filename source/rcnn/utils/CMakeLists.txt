add_library(utils STATIC registry.cpp tovec.cpp timer.cpp bisect.cpp)
target_include_directories(utils 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/utils/)
target_link_libraries(utils PRIVATE modeling)