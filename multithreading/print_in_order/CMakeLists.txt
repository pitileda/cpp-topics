cmake_minimum_required(VERSION 3.0.0)
project(PrintInOrder VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(PrintInOrder main.cpp)
target_include_directories(PrintInOrder PRIVATE include)
target_compile_features(PrintInOrder PRIVATE cxx_std_20)
target_compile_options(PrintInOrder PRIVATE -pthread)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
