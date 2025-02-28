#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iostream>

int WIDTH = 800;
int HEIGHT = 600;
int MAX_ITER = 1000;

double centerX = -0.5;
double centerY = 0.0;
double zoom = 1.0;

__global__ void compute_mandelbrot(unsigned char* image, double centerX,
                                   double centerY, double zoom, int width,
                                   int height, int max_iter) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = width * height;

  if (idx < total_pixels) {
    int x = idx % width;
    int y = idx / width;

    // Scale x and y to the Mandelbrot coordinate space with zoom and pan
    double scale = 1.5 / zoom;
    double real = (x - width / 2.0) * scale / (width / 2.0) + centerX;
    double imag = (y - height / 2.0) * scale / (height / 2.0) + centerY;

    double c_real = real;
    double c_imag = imag;

    int iteration = 0;

    while (real * real + imag * imag <= 4.0 && iteration < max_iter) {
      double temp_real = real * real - imag * imag + c_real;
      imag = 2.0 * real * imag + c_imag;
      real = temp_real;
      iteration++;
    }

    // Map the number of iterations to a grayscale value
    image[idx] = static_cast<unsigned char>(255 * iteration / max_iter);
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

  // Parameters for animation
  int total_frames = 120;  // 4 seconds at 30 fps
  double zoom_start = 1.0;
  double zoom_end = 1000.0;

  for (int frame = 0; frame < total_frames; frame++) {
    // Calculate zoom for the current frame
    double t = static_cast<double>(frame) / total_frames;
    zoom = zoom_start * pow(zoom_end / zoom_start, t);

    // Launch the kernel with current parameters
    compute_mandelbrot<<<blocks, threads>>>(d_image, centerX, centerY, zoom,
                                            WIDTH, HEIGHT, MAX_ITER);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_image, d_image, image_bytes, cudaMemcpyDeviceToHost);

    // Generate filename
    char filename[256];
    sprintf(filename, "frame_%04d.pgm", frame);

    // Save the image as a PGM file
    std::ofstream img(filename);
    img << "P2\n" << WIDTH << " " << HEIGHT << "\n255\n";
    for (int i = 0; i < image_size; i++) {
      img << static_cast<int>(h_image[i]) << " ";
      if ((i + 1) % WIDTH == 0) img << "\n";
    }
    img.close();

    std::cout << "Generated frame: " << filename << "\n";
  }

  // Free device memory
  cudaFree(d_image);

  // Free host memory
  delete[] h_image;

  std::cout << "Animation frames generated.\n";
  return 0;
}
