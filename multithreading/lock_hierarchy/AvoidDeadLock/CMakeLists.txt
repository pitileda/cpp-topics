cmake_minimum_required(VERSION 3.5)

project(AvoidDeadLock LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads)

add_executable(AvoidDeadLock avoid_deadlock_hierarchy.cc)
target_include_directories(AvoidDeadLock PRIVATE .)
target_link_libraries(AvoidDeadLock ${CMAKE_THREAD_LIBS_INIT})
