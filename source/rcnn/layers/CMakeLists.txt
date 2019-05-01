if(DEFINED ENV{CUDA_HOME})
  file(GLOB sources "cuda/*.cu" "*.cpp")
else()
  file(GLOB sources "cpu/*.cpp" "*.cpp")
endif()

add_library(layers ${sources})
if(DEFINED ENV{CUDA_HOME})
  target_compile_definitions(layers PRIVATE WITH_CUDA=ON)
  target_include_directories(layers 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/ 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/cuda )
else()
  target_include_directories(layers 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/ 
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../include/rcnn/layers/cpu )
  message("no cuda")
endif()


target_link_libraries(layers "${TORCH_LIBRARIES}")