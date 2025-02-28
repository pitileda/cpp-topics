#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1024
#define HEIGHT 768
#define MAX_ITER 1000

// CUDA kernel: Computes a Mandelbrot-like set into an RGBA pixel buffer.
__global__ void mandelbrot_kernel(unsigned char* d_ptr, int width, int height,
                                  float zoom, float offsetX, float offsetY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  // Map pixel coordinate to the complex plane.
  float jx = (float)(x - width / 2) / (0.5f * zoom * width) + offsetX;
  float jy = (float)(y - height / 2) / (0.5f * zoom * height) + offsetY;

  float zx = 0.0f, zy = 0.0f;
  int i = 0;
  while (zx * zx + zy * zy < 4.0f && i < MAX_ITER) {
    float tmp = zx * zx - zy * zy + jx;
    zy = 2.0f * zx * zy + jy;
    zx = tmp;
    i++;
  }

  // Write color to RGBA pixel buffer (4 bytes per pixel).
  int pixel_index = (y * width + x) * 4;
  if (i == MAX_ITER) {
    // Inside the set: black.
    d_ptr[pixel_index + 0] = 0;
    d_ptr[pixel_index + 1] = 0;
    d_ptr[pixel_index + 2] = 0;
    d_ptr[pixel_index + 3] = 255;
  } else {
    // Outside: use a color gradient.
    unsigned char color = (unsigned char)(255 * i / MAX_ITER);
    d_ptr[pixel_index + 0] = color;
    d_ptr[pixel_index + 1] = 0;
    d_ptr[pixel_index + 2] = 255 - color;
    d_ptr[pixel_index + 3] = 255;
  }
}

__global__ void test_kernel(unsigned char* d_ptr, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;
  int pixel_index = (y * width + x) * 4;
  d_ptr[pixel_index + 0] = 255;  // R
  d_ptr[pixel_index + 1] = 0;    // G
  d_ptr[pixel_index + 2] = 0;    // B
  d_ptr[pixel_index + 3] = 255;  // A
}

int main() {
  // --- Removed erroneous initial kernel call ---

  // Initialize GLFW for creating a window and OpenGL context.
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    return -1;
  }

  // Request OpenGL 3.3 core profile.
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window =
      glfwCreateWindow(WIDTH, HEIGHT, "CUDA Mandelbrot Animation", NULL, NULL);
  if (!window) {
    fprintf(stderr, "Failed to create GLFW window\n");
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  cudaGLSetGLDevice(0);

  // Initialize GLEW.
  glewExperimental = GL_TRUE;
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    return -1;
  }

  // Create an OpenGL Pixel Buffer Object (PBO) to store image data.
  GLuint pbo;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH * HEIGHT * 4, NULL,
               GL_DYNAMIC_DRAW);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

  // Register the PBO with CUDA.
  cudaGraphicsResource* cuda_pbo_resource;
  cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo,
                               cudaGraphicsMapFlagsWriteDiscard);

  // Create an OpenGL texture to display the PBO data.
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Set up a full-screen quad (two triangles) with positions and texture
  // coordinates.
  float vertices[] = {// positions    // tex coords
                      -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,  -1.0f, 1.0f, 0.0f,
                      1.0f,  1.0f,  1.0f, 1.0f, -1.0f, 1.0f,  0.0f, 1.0f};
  unsigned int indices[] = {0, 1, 2, 2, 3, 0};

  GLuint VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // Position attribute.
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // Texture coordinate attribute.
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Simple vertex and fragment shaders to render the texture.
  const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoord;
    out vec2 TexCoord;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        TexCoord = aTexCoord;
    }
  )";

  const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D screenTexture;
    void main() {
        FragColor = texture(screenTexture, TexCoord);
    }
  )";

  // Compile vertex shader.
  GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    fprintf(stderr, "Vertex Shader compilation failed: %s\n", infoLog);
  }

  // Compile fragment shader.
  GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    fprintf(stderr, "Fragment Shader compilation failed: %s\n", infoLog);
  }

  // Link shaders into a program.
  GLuint shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    fprintf(stderr, "Shader Program linking failed: %s\n", infoLog);
  }
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // Main loop: update animation and display it.
  while (!glfwWindowShouldClose(window)) {
    // Map the PBO so CUDA can write into it.
    unsigned char* d_ptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_ptr, &num_bytes,
                                         cuda_pbo_resource);

    // Use the GLFW time to animate parameters.
    float time = (float)glfwGetTime();
    // float zoom = 1.0f + 0.5f * sin(time);
    // float offsetX = -0.5f + 0.5f * cos(time);
    // float offsetY = 0.0f + 0.5f * sin(time * 0.5f);
    float zoom = 0.5f;
    float offsetX = -0.7f;
    float offsetY = 0.0f;

    // Launch the CUDA kernel.
    dim3 block(16, 16);
    dim3 grid((WIDTH + block.x - 1) / block.x,
              (HEIGHT + block.y - 1) / block.y);
    // mandelbrot_kernel<<<grid, block>>>(d_ptr, WIDTH, HEIGHT, zoom, offsetX,
    //                                    offsetY);
    // cudaDeviceSynchronize();
    mandelbrot_kernel<<<grid, block>>>(d_ptr, WIDTH, HEIGHT, zoom, offsetX,
                                       offsetY);
    cudaDeviceSynchronize();
    cudaError_t last_err = cudaGetLastError();
    if (last_err != cudaSuccess) {
      fprintf(stderr, "Kernel launch error: %s\n",
              cudaGetErrorString(last_err));
    }

    cudaError_t map_err = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
    if (map_err != cudaSuccess) {
      fprintf(stderr, "cudaGraphicsMapResources error: %s\n",
              cudaGetErrorString(map_err));
    }
    cudaError_t err = cudaGraphicsResourceGetMappedPointer(
        (void**)&d_ptr, &num_bytes, cuda_pbo_resource);
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaGraphicsResourceGetMappedPointer error: %s\n",
              cudaGetErrorString(err));
    }

    // Unmap the PBO.
    cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

    // Update the texture with the new data from the PBO.
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA,
                    GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // Render the full-screen quad.
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(shaderProgram);
    int loc = glGetUniformLocation(shaderProgram, "screenTexture");
    glUniform1i(loc, 0);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // Swap buffers and poll events.
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Cleanup resources.
  cudaGraphicsUnregisterResource(cuda_pbo_resource);
  glDeleteBuffers(1, &pbo);
  glDeleteTextures(1, &texture);
  glDeleteProgram(shaderProgram);
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
