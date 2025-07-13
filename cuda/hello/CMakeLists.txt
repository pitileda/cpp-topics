# in bashrc/zshrc need to set env
#export PATH=/usr/local/cuda-12.3/bin${PATH:+:${PATH}}
#export PATH=/usr/local/cuda-12.3/include${PATH:+:${PATH}}

cmake_minimum_required(VERSION 3.8)

project(cuda_debug_example)
enable_language("CUDA")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# set(CMAKE_CUDA_ARCHITECTURES "sm_86" CACHE STRING "CUDA architectures" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/hello.cu)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES native)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

add_executable(${PROJECT_NAME}-simple ${PROJECT_SOURCE_DIR}/hello_simple.cu)
set_target_properties(${PROJECT_NAME}-simple PROPERTIES CUDA_ARCHITECTURES 86)
target_include_directories(${PROJECT_NAME}-simple PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME}-simple PRIVATE CUDA::cudart)