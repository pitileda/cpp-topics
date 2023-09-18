# Inject experimental variables because they must be normal variables
set(CMAKE_PROJECT_INCLUDE Experimental.cmake CACHE STRING "")

set(CMAKE_BUILD_TYPE Release CACHE STRING "")

set(CMAKE_CXX_STANDARD 20 CACHE STRING "")
# Default to C++ extensions being off. Clang's modules support have trouble
# with extensions right now and it is not required for any other compiler
set(CMAKE_CXX_EXTENSIONS OFF CACHE STRING "")
