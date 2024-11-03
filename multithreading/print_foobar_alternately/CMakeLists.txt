cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(print_foobar VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
find_package(Threads)

add_executable(print_foobar print_foobar_alternately_sem.cc)
target_link_libraries(print_foobar ${CMAKE_THREAD_LIBS_INIT})