add_library(solver STATIC build.cpp)
target_include_directories(solver 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/solver/)
target_link_libraries(structures PUBLIC ${TORCH_LIBRARIES})