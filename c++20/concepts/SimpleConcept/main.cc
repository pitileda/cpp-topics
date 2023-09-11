#include <concepts>
#include <iostream>
#include <ostream>
#include <type_traits>

template <typename T>
concept Number = std::is_floating_point_v<T> || std::is_integral_v<T>;

template <Number N>
auto sum(N x, N y) -> decltype(x + y) {
  return x + y;
};

int main(int argc, char const *argv[]) {
  auto a = sum(1, 2);
  auto b = sum(1.3, 2.4);
  std::cout << a << std::endl << b << std::endl;
  return 0;
}
