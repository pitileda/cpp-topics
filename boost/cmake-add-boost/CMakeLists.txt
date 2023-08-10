cmake_minimum_required(VERSION 3.14)
project(BoostExample)

find_package(Boost COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(boost_example main.cc)
target_link_libraries(boost_example PRIVATE Boost::filesystem)