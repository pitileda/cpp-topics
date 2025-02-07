#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define N 4'000'000'000

float add(float* out, float* a, float* b, uint64_t n) {
  float res = 0;
  for (uint64_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
    res += out[i];
  }
  return res;
};

int main() {
  float *a, *b, *out;
  a = (float*)malloc(sizeof(float) * N);
  b = (float*)malloc(sizeof(float) * N);
  out = (float*)malloc(sizeof(float) * N);

  // float in1, in2;
  // std::cin >> in1;
  // std::cin >> in2;

  for (uint64_t i = 0; i <= N; ++i) {
    a[i] = 12.3;
    b[i] = 13.4;
  }

  printf("%f", add(out, a, b, N));
  return 0;
}
