file(GLOB_RECURSE sources "*.cpp")

add_library(data STATIC ${sources})
target_include_directories(data 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/data/ ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/rapidjson/include)
target_link_libraries(data PUBLIC ${TORCH_LIBRARIES} structures cocotool utils config ${OpenCV_LIBS})
