# Target name
set(target UnitTests)

# message(${GTEST_INCLUDE_DIRS})
find_package(GTest REQUIRED)

# Sources
file(GLOB_RECURSE sources
    *.cpp)

# Build executable
add_executable(${target}
    test.cpp
    ${sources})

target_include_directories(${target} PUBLIC ${gtest_SOURCE_DIR}/include ${gmock_SOURCE_DIR}/include )

# Link libraries
target_link_libraries(${target}
    PUBLIC
    gtest
    maskrcnn
    gtest_main
    ${TORCH_LIBRARIES})

add_test(NAME ${target}
  COMMAND ${target}
)