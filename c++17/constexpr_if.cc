template <int N>
constexpr int fibonacci() {
  if constexpr (N >= 2) {
    return fibonacci<N - 1>() + fibonacci<N - 2>();
  }

  return N;
}

template <>
constexpr int fibonacci<1>() {
  return 1;
}

template <>
constexpr int fibonacci<0>() {
  return 1;
}

int main() { return fibonacci<6>(); }