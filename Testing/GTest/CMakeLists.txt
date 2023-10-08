cmake_minimum_required(VERSION 3.25 FATAL_ERROR)
project(GTestHello VERSION 0.1.0 LANGUAGES CXX)

# set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
	googletest
	URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

enable_testing()

# Set the common compiler flags, including --coverage
# if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
#     message("Enabling code coverage analysis")
#     set(CMAKE_CXX_FLAGS "-g -O0 -Wall -fprofile-arcs -ftest-coverage")
# 		set(CMAKE_C_FLAGS "-g -O0 -Wall -W -fprofile-arcs -ftest-coverage")
# 		set(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
# endif()

add_executable(${PROJECT_NAME} hello_test.cc)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_link_libraries(${PROJECT_NAME} GTest::gtest_main)
# Add the coverage flags to the main project executable as well
if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    target_compile_options(${PROJECT_NAME} PRIVATE --coverage)
endif()

add_executable(FifoTest test/TestFifo.cc)
target_include_directories(FifoTest PRIVATE include)
target_link_libraries(FifoTest GTest::gtest_main)
# Add the coverage flags to the main project executable as well
if (CMAKE_BUILD_TYPE STREQUAL "Coverage")
    target_compile_options(FifoTest PRIVATE --coverage)
endif()

add_executable(StaticTest test/StaticClass.cc)
target_include_directories(StaticTest PRIVATE include)
target_link_libraries(StaticTest GTest::gtest_main GTest::gmock GTest::gmock_main gcov)
target_compile_options(StaticTest PUBLIC --coverage)

option(WITH_UNIT_TESTS "building with UNIT_TESTS" ON)
if(WITH_UNIT_TESTS)
	add_definitions(-DUNIT_TESTS)
	SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
	SET(GCC_COVERAGE_LINK_FLAGS    "-lgcov")
	SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")
	SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
endif()

add_library(bar src/bar.cc)
target_include_directories(bar PRIVATE include)

add_executable(BarTest test/bar_test.cc)
target_include_directories(BarTest PRIVATE include)
target_link_libraries(BarTest GTest::gtest_main GTest::gmock GTest::gmock_main gcov bar)
target_compile_options(BarTest PUBLIC --coverage)

add_test(NAME Test1 COMMAND StaticTest)