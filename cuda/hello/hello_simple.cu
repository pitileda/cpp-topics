#include <cstdio>

__global__ void cuda_hello() { printf("Hello from GPU"); }

int main() {
  cuda_hello<<<1, 1>>>();
  return 0;
}
