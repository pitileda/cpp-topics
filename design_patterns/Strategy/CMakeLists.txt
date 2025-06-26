cmake_minimum_required(VERSION 3.31)
project(Strategy)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Strategy main.cpp)
target_include_directories(Strategy PRIVATE tests include)
target_link_libraries(Strategy PRIVATE GTest::gtest_main GTest::gmock)
include(GoogleTest)
gtest_discover_tests(Strategy)
