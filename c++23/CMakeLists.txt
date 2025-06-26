cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

set(CMAKE_C_COMPILER "/home/oem/tmp/llvm-project/build/bin/clang")
set(CMAKE_CXX_COMPILER "/home/oem/tmp/llvm-project/build/bin/clang++")

project(CPP23 VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # extesnsions is off in std.pcm

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodule-file=std=/home/oem/tmp/llvm-project/build/std.pcm")

add_executable(hello hello_cpp23.cc)
target_compile_features(hello PRIVATE cxx_std_23)

add_executable(expected expected.cc)
target_compile_features(expected PRIVATE cxx_std_23)

add_executable(print print.cc)
target_compile_features(print PRIVATE cxx_std_23)

add_executable(generator generator.cc)
target_compile_features(generator PRIVATE cxx_std_23)

add_executable(flat_associative_containers flat_associative_containers.cc)
target_compile_features(flat_associative_containers PRIVATE cxx_std_23)