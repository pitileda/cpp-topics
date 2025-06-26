cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(concepts VERSION 0.1.0 LANGUAGES CXX C)
add_executable(${PROJECT_NAME} concepts.cc)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)