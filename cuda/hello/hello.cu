#include <stdio.h>

#include <iostream>

#include "cuda_runtime.h"

void print_CUDA_version() {
  std::cout << "__CUDACC_VER_MAJOR__: " << __CUDACC_VER_MAJOR__ << "\n"
            << "__CUDACC_VER_MINOR__" << __CUDACC_VER_MINOR__ << "\n"
            << "__CUDACC_VER_BUILD__: " << __CUDACC_VER_BUILD__ << "\n\n";

  int runtime_ver;
  cudaRuntimeGetVersion(&runtime_ver);
  std::cout << "CUDA Runtime version: " << runtime_ver << "\n";

  int driver_ver;
  cudaDriverGetVersion(&driver_ver);
  std::cout << "CUDA Driver version: " << driver_ver << "\n";
}

__global__ void hello() {
  printf("Hello from GPU\n");
  printf("Bye!\n");
  int x = 5;
  x++;
  printf("The x: %i", x);
}

int main() {
  hello<<<1, 100>>>();

  std::cout << "Running CUDA.\n";
  print_CUDA_version();
  cudaDeviceSynchronize();
  return 0;
}
