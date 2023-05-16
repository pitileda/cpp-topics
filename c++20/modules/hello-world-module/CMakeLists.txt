cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR)

project(hello LANGUAGES CXX)

# set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP 1)
# turn on the experimental API
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API  "2182bf5c-ef0d-489a-91da-49dbc3090d2a")

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  include(clang_modules.cmake)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  include(gcc_modules.cmake)
endif()

set(CMAKE_CXX_STANDARD 20)# turn on the dynamic depends for ninja

add_library(foo)
target_sources(foo
	PUBLIC
		FILE_SET CXX_MODULES FILES
		foo.cc
)

add_executable(hello main.cc)