cmake_minimum_required(VERSION 3.15)
project(simpleConcept)
add_executable(${PROJECT_NAME} main.cc)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)