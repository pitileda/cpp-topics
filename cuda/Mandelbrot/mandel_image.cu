#include <cuda_runtime.h>

#include <fstream>
#include <iostream>

const int WIDTH = 800;
const int HEIGHT = 600;
const int MAX_ITER = 1000;

__global__ void compute_mandelbrot(unsigned char* image) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = WIDTH * HEIGHT;

  if (idx < total_pixels) {
    int x = idx % WIDTH;
    int y = idx / WIDTH;

    // Scale x and y to the Mandelbrot coordinate space
    double real = (x - WIDTH / 2.0) * 3.5 / WIDTH;
    double imag = (y - HEIGHT / 2.0) * 2.0 / HEIGHT;

    double c_real = real;
    double c_imag = imag;

    int iteration = 0;
    const int max_iteration = MAX_ITER;

    while (real * real + imag * imag <= 4.0 && iteration < max_iteration) {
      double temp_real = real * real - imag * imag + c_real;
      imag = 2.0 * real * imag + c_imag;
      real = temp_real;
      iteration++;
    }

    // Map the number of iterations to a grayscale value
    image[idx] = static_cast<unsigned char>(255 * iteration / MAX_ITER);
  }
}

int main() {
  const int image_size = WIDTH * HEIGHT;
  const int image_bytes = image_size * sizeof(unsigned char);

  // Allocate memory on the host
  unsigned char* h_image = new unsigned char[image_size];

  // Allocate memory on the device
  unsigned char* d_image;
  cudaMalloc(&d_image, image_bytes);

  // Define block and grid sizes
  const int threads = 256;
  const int blocks = (image_size + threads - 1) / threads;

  // Launch the kernel
  compute_mandelbrot<<<blocks, threads>>>(d_image);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Copy the result back to the host
  cudaMemcpy(h_image, d_image, image_bytes, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_image);

  // Save the image as a PGM file
  std::ofstream img("mandelbrot.pgm");
  img << "P2\n" << WIDTH << " " << HEIGHT << "\n255\n";
  for (int i = 0; i < image_size; i++) {
    img << static_cast<int>(h_image[i]) << " ";
    if ((i + 1) % WIDTH == 0) img << "\n";
  }
  img.close();

  // Free host memory
  delete[] h_image;

  std::cout << "Mandelbrot image generated: mandelbrot.pgm\n";
  return 0;
}
