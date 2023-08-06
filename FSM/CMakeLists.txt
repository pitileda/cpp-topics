cmake_minimum_required(VERSION 3.20)

project(FSM DESCRIPTION "Finite State Machine" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(usingMap usingMap.cpp)
add_executable(usingSwitch usingSwitch.cpp)

target_include_directories(usingMap PRIVATE map)
target_include_directories(usingSwitch PRIVATE switch)