cmake_minimum_required(VERSION 3.31)
project(Builder)

set(CMAKE_CXX_STANDARD 20)

add_executable(fluent_builder fluent_builder.cpp)

add_executable(Builder facet_builder.cpp
        PersonBuilder.cpp
        Person.cpp
)
target_include_directories(Builder PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})