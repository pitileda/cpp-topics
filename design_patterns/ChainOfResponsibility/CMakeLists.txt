cmake_minimum_required(VERSION 3.31)
project(ChainOfResponsibility)

set(CMAKE_CXX_STANDARD 20)

add_executable(ChainOfResponsibility pointer_chain.cpp)
add_executable(ChainList list_pointer_chain.cpp)
