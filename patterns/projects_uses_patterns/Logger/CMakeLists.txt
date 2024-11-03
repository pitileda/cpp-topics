cmake_minimum_required(VERSION 3.18)

project(Logger VERSION 0.1.0 LANGUAGES CXX C)

add_executable(Logger)
# add_library(logger )

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
target_sources(${CMAKE_PROJECT_NAME} PRIVATE log.cc)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(src)