cmake_minimum_required(VERSION 3.15)
project(CamelCase)

if(POLICY CMP0135)
	cmake_policy(SET CMP0135 NEW)
	set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(${PROJECT_NAME} camelcase.cc)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} GTest::gtest_main)

add_executable(${PROJECT_NAME}Tests tests.cc)
target_include_directories(${PROJECT_NAME}Tests PRIVATE include)
target_link_libraries(${PROJECT_NAME}Tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME})