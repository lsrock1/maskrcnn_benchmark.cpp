# Target name
set(target UnitTests)

# Sources
file(GLOB sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.cpp)

# Build executable
add_executable(${target}
    test.cpp
    ${sources})

# Link libraries
target_link_libraries(${target}
    PRIVATE
    gtest
    maskrcnn
    ${TORCH_LIBRARIES})

gtest_add_tests(
    TARGET ${target}
)