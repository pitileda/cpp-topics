cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)
project(CPP23 VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_COMPILER "/home/oem/tmp/llvm-project/build/bin/clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodule-file=std=/home/oem/tmp/llvm-project/build/std.pcm")

add_executable(hello hello_cpp23.cc)
target_compile_features(hello PRIVATE cxx_std_23)