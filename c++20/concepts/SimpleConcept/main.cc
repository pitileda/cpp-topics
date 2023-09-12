#include <concepts>
#include <iostream>
#include <type_traits>

template <typename T>
concept NumberType = std::is_floating_point_v<T> || std::is_integral_v<T>;

template <NumberType Number>
auto sum(Number x, Number y) -> decltype(x + y) {
  return x + y;
};

int main(int argc, char const *argv[]) {
  auto a = sum(1, 2);
  auto b = sum(1.3, 2.4);
  std::cout << a << std::endl << b << std::endl;
  // no operand of the disjunction is satisfied
  // std::cout << sum("hi", "ho") << std::endl;
  return 0;
}
