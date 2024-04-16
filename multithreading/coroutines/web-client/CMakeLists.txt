cmake_minimum_required(VERSION 3.26)
project(web-client)

find_package(CURL REQUIRED)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} main.cc)
target_link_libraries(${PROJECT_NAME} CURL::libcurl ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${PROJECT_NAME} PUBLIC header)
