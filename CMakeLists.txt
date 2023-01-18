cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(Topics
    DESCRIPTION
        "C++ topics"
    VERSION
        0.1
    LANGUAGES
        CXX)

# Download and build gtest
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

FetchContent_MakeAvailable(googletest)

add_subdirectory(./destructor)

enable_testing()
