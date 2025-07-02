cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(FunctionTest VERSION 0.1.0 LANGUAGES CXX C)

enable_testing()
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

add_executable(${PROJECT_NAME} test_fun.cc)
target_link_libraries(${PROJECT_NAME} GTest::gtest GTest::gtest_main)
target_compile_options(${PROJECT_NAME} PRIVATE
	-Werror -Wall -Wextra -Wpedantic -Wshadow -Wconversion)

add_executable(${PROJECT_NAME}-param test_fun_param.cc)
target_link_libraries(${PROJECT_NAME}-param GTest::gtest GTest::gtest_main)
target_compile_options(${PROJECT_NAME}-param PRIVATE
	-Werror -Wall -Wextra -Wpedantic -Wshadow -Wconversion)

add_test(NAME FunctionTest COMMAND ${PROJECT_NAME})
add_test(NAME FunctionParamTest COMMAND ${PROJECT_NAME}-param)