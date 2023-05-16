cmake_minimum_required(VERSION 3.0.0)
project(trie VERSION 0.1.0)

include(CTest)
enable_testing()

# Set default sanitizer flags
set(SANITIZER_FLAGS "")

# Check if GCC is being used
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(STATUS "Using GCC compiler")
    set(SANITIZER_FLAGS "-fsanitize=address")
endif()

# Check if Clang is being used
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Using Clang compiler")
    set(SANITIZER_FLAGS "-fsanitize=leak")
endif()

# Set Google Test download and install options
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests since we're pulling in Google Test" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON CACHE BOOL "Downloading Google Benchmark's dependencies (including Google Test)")

include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.1)

FetchContent_MakeAvailable(benchmark)

add_executable(trie main.cc)

target_compile_features(trie PRIVATE cxx_std_20)
target_include_directories(trie PRIVATE include)
target_sources(trie PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/trie.cc)

add_subdirectory(test)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
