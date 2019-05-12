set(target maskrcnn)
 
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lib/libtorch)
find_package(CUDA)

if(CUDA_FOUND)
  message("CUDA FOUND")
  enable_language(CUDA)
  string(APPEND CMAKE_CUDA_FLAGS "-DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__")
endif()

add_subdirectory(rcnn/config)
# add_subdirectory(rcnn/data)
add_subdirectory(rcnn/layers)
add_subdirectory(rcnn/modeling)
add_subdirectory(rcnn/structures)
# add_subdirectory(rcnn/utils)
# add_subdirectory(tools)

add_library(${target} INTERFACE)
target_link_libraries(${target} INTERFACE modeling config layers structures)