#include <cstdio>

__global__ void dummy_kernel(int *out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  out[idx] = idx;
}

int main() {
  const int N = 8;
  int h_out[N] = {0};
  int *d_out;

  cudaMalloc(&d_out, N * sizeof(int));
  dummy_kernel<<<2, 4>>>(d_out);

  cudaError_t launch_err = cudaGetLastError();
  if (launch_err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(launch_err));
    return 1;
  }

  cudaError_t sync_err = cudaDeviceSynchronize();
  if (sync_err != cudaSuccess) {
    printf("Kernel sync error: %s\n", cudaGetErrorString(sync_err));
    return 1;
  }

  cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; ++i) printf("h_out[%d] = %d\n", i, h_out[i]);

  cudaFree(d_out);
  return 0;
}
