cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(Coro_basic VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} coro_basic.cc)