cmake_minimum_required(VERSION 3.22 FATAL_ERROR)

project(MyProj VERSION 0.1.0 LANGUAGES CXX)

file(GLOB MYLIB_SOURCES "libs/*.cc" "libs/*.h")
add_library(my ${MYLIB_SOURCES})

add_executable(app main.cc)

# target_include_directories(app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/libs)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/libs)
find_package(Boost COMPONENTS system filesystem)

include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(app PRIVATE my Boost::filesystem)
