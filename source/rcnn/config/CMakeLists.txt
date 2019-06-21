find_package(yaml-cpp REQUIRED)

add_library(config STATIC defaults.cpp paths_catalog.cpp)
target_include_directories(config 
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/config/ 
  ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(config PUBLIC ${YAML_CPP_LIBRARIES} utils)
