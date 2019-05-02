if(CUDA_FOUND)
  file(GLOB sources "cuda/*.cu" "cpu/*.cpp" "*.cpp" )
else()
  file(GLOB sources "cpu/*.cpp" "*.cpp")
endif()

add_library(layers ${sources})
if(CUDA_FOUND)
  target_compile_definitions(layers PUBLIC WITH_CUDA=ON)
  target_include_directories(layers 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/ 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/cpu
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/cuda )
else()
  target_include_directories(layers 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/ 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/cpu )
endif()

target_link_libraries(layers "${TORCH_LIBRARIES}")