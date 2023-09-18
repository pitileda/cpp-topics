cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(hello LANGUAGES CXX)

set(CMAKE_EXPERIMENTAL_CXX_MODULE_DYNDEP ON)
# turn on the experimental API
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API aa1f7df0-828a-4fcd-9afc-2dc80491aca7)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  include(clang_modules.cmake)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  include(gcc_modules.cmake)
endif()

set(CMAKE_CXX_STANDARD 20)# turn on the dynamic depends for ninja

add_library(foo)
target_sources(foo PUBLIC main.cc)
target_sources(foo
	PRIVATE
		FILE_SET foo_module TYPE CXX_MODULES 
		FILES foo.cc
)