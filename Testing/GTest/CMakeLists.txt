cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(GTestHello VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)


enable_testing()
add_executable(${PROJECT_NAME} hello_test.cc)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} GTest::gtest_main)

add_executable(FifoTest test/TestFifo.cc)
target_include_directories(FifoTest PRIVATE include)
target_link_libraries(FifoTest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})