#include <chrono>
#include <cstdint>
#include <iostream>

// Number of iterations for the benchmark
const size_t ITERATIONS = 100'000'000;

// Volatile variables to prevent optimization
volatile int int_sink = 0;
volatile double float_sink = 0.0;

// Benchmark Integer Addition
void benchmark_int_addition() {
  int a = 1, b = 2, c = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += a + b;
  }
  auto end = std::chrono::high_resolution_clock::now();
  int_sink = c;  // Use the result
  std::chrono::duration<double> duration = end - start;
  std::cout << "Integer Addition Time: " << duration.count() << " seconds\n";
}

// Benchmark Floating-Point Addition
void benchmark_float_addition() {
  double a = 1.0, b = 2.0, c = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += a + b;
  }
  auto end = std::chrono::high_resolution_clock::now();
  float_sink = c;  // Use the result
  std::chrono::duration<double> duration = end - start;
  std::cout << "Floating-Point Addition Time: " << duration.count()
            << " seconds\n";
}

// Benchmark Integer Multiplication
void benchmark_int_multiplication() {
  int a = 3, b = 4;
  int64_t c = 0;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += a * b;
  }
  auto end = std::chrono::high_resolution_clock::now();
  int_sink = c;  // Use the result
  std::chrono::duration<double> duration = end - start;
  std::cout << "Integer Multiplication Time: " << duration.count()
            << " seconds\n";
}

// Benchmark Floating-Point Multiplication
void benchmark_float_multiplication() {
  double a = 3.0, b = 4.0, c = 0.0;
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += a * b;
  }
  auto end = std::chrono::high_resolution_clock::now();
  float_sink = c;  // Use the result
  std::chrono::duration<double> duration = end - start;
  std::cout << "Floating-Point Multiplication Time: " << duration.count()
            << " seconds\n";
}

// Volatile variables to prevent compiler optimizations
volatile uint32_t and_sink = 0;
volatile uint32_t or_sink = 0;
volatile uint32_t xor_sink = 0;
volatile uint32_t not_sink = 0;

// Benchmark Bitwise AND
void benchmark_bitwise_and() {
  uint32_t a = 0xAAAAAAAA;  // 10101010...
  uint32_t b = 0x55555555;  // 01010101...
  uint32_t c = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += (a & b);
  }
  auto end = std::chrono::high_resolution_clock::now();
  and_sink = c;  // Use the result to prevent optimization

  std::chrono::duration<double> duration = end - start;
  std::cout << "Bitwise AND Time: " << duration.count() << " seconds\n";
}

// Benchmark Bitwise OR
void benchmark_bitwise_or() {
  uint32_t a = 0xAAAAAAAA;  // 10101010...
  uint32_t b = 0x55555555;  // 01010101...
  uint32_t c = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += (a | b);
  }
  auto end = std::chrono::high_resolution_clock::now();
  or_sink = c;  // Use the result to prevent optimization

  std::chrono::duration<double> duration = end - start;
  std::cout << "Bitwise OR Time: " << duration.count() << " seconds\n";
}

// Benchmark Bitwise XOR
void benchmark_bitwise_xor() {
  uint32_t a = 0xAAAAAAAA;  // 10101010...
  uint32_t b = 0x55555555;  // 01010101...
  uint32_t c = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += (a ^ b);
  }
  auto end = std::chrono::high_resolution_clock::now();
  xor_sink = c;  // Use the result to prevent optimization

  std::chrono::duration<double> duration = end - start;
  std::cout << "Bitwise XOR Time: " << duration.count() << " seconds\n";
}

// Benchmark Bitwise NOT
void benchmark_bitwise_not() {
  uint32_t a = 0xAAAAAAAA;  // 10101010...
  uint32_t c = 0;

  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < ITERATIONS; ++i) {
    c += (~a);
  }
  auto end = std::chrono::high_resolution_clock::now();
  not_sink = c;  // Use the result to prevent optimization

  std::chrono::duration<double> duration = end - start;
  std::cout << "Bitwise NOT Time: " << duration.count() << " seconds\n";
}

int main() {
  std::cout << "Benchmarking Integer vs Floating-Point Operations\n\n";

  benchmark_int_addition();
  benchmark_float_addition();
  benchmark_int_multiplication();
  benchmark_float_multiplication();

  // Prevent the compiler from optimizing away the sink variables
  std::cout << "Sinks: " << int_sink << " " << float_sink << "\n";

  std::cout << "Benchmarking Bitwise Logical Operations with " << ITERATIONS
            << " iterations each.\n\n";

  benchmark_bitwise_and();
  benchmark_bitwise_or();
  benchmark_bitwise_xor();
  benchmark_bitwise_not();

  // Prevent the compiler from optimizing away the sink variables
  std::cout << "\nSinks: " << and_sink << " " << or_sink << " " << xor_sink
            << " " << not_sink << "\n";

  return 0;
}
