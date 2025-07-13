cmake_minimum_required(VERSION 3.18)
project(CudaApp LANGUAGES CXX CUDA)
enable_language(CUDA)

# Enable standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard (optional)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)

# Enable separable compilation (recommended for larger apps)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Add executable
cuda_add_executable(Sum two_arrays_sum.cu)

# Ensure target uses CUDA
set_target_properties(Sum PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "native"
)
# target_link_libraries(Sum PRIVATE cuda)

# Optional: verbose CUDA flags for debugging
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall")
