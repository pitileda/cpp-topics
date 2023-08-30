cmake_minimum_required(VERSION 3.15)
project(onTimeoutForBlockingCall)
find_package(Threads REQUIRED)
add_executable(forked forked.cc)
target_link_libraries(forked ${CMAKE_THREAD_LIBS_INIT})

add_executable(threaded threaded.cc)
target_link_libraries(threaded ${CMAKE_THREAD_LIBS_INIT})
