set(target maskrcnn)

add_subdirectory(coco)
add_subdirectory(rcnn/config)
add_subdirectory(rcnn/data)
add_subdirectory(rcnn/layers)
add_subdirectory(rcnn/modeling)
add_subdirectory(rcnn/structures)
add_subdirectory(rcnn/utils)
# add_subdirectory(tools)

add_library(${target} INTERFACE)
target_link_libraries(${target} INTERFACE modeling config layers structures utils cocotool data)