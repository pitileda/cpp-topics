cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)
project(strategy VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(strategy main.cpp)
target_include_directories(strategy PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(strategy PUBLIC cxx_inheriting_constructors)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

#add integration tests
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/integration/tests.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/unit.cmake)

enable_testing()