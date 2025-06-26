cmake_minimum_required(VERSION 3.31)
project(Flyweight)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(Flyweight main.cpp)
add_executable(${PROJECT_NAME}Test test.cpp)
target_include_directories(${PROJECT_NAME}Test PRIVATE tests include)
target_link_libraries(${PROJECT_NAME}Test GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}Test)