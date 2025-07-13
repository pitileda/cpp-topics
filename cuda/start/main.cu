#include <cstdio>

__global__ void test_kernel(int *data) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  data[idx] = idx;
}

int main() {
  int *d_data;
  cudaMalloc(&d_data, 8 * sizeof(int));

  test_kernel<<<2, 4>>>(d_data);

  cudaError_t launch_err = cudaGetLastError();
  printf("Kernel launch: %s\n", cudaGetErrorString(launch_err));

  cudaError_t sync_err = cudaDeviceSynchronize();
  printf("Kernel sync: %s\n", cudaGetErrorString(sync_err));

  cudaFree(d_data);
  return 0;
}
