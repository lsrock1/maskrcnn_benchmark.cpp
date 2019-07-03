add_library(engine STATIC trainer.cpp inference.cpp parallel.cpp)
target_include_directories(engine 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/engine/ ${OpenCV_INCLUDE_DIRS})
target_link_libraries(engine modeling utils data config solver structures ${OpenCV_LIBS})