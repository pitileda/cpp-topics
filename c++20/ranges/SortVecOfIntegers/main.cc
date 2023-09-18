#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>

template <typename Container>
auto print(const Container& container) {
  bool isFirst = true;
  for (const auto& item : container) {
    if (isFirst) {
      std::cout << item;
      isFirst = false;
    }
    std::cout << ", " << item;
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  std::vector vec{-1, 2, 6, -3, 5, -8};
  std::ranges::sort(vec);
  print(vec);
  return 0;
}
