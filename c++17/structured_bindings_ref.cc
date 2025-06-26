#include <iostream>

int main() {
  long long theArray[4] = {1ull, 2ull, 3ull, 4ull};

  // get the ref-s
  auto& [a, b, c, d] = theArray;
  b = 7ull;
  for (const auto& el : theArray) {
    std::cout << el << std::endl;
  }
  return static_cast<int>(theArray[3]);
}