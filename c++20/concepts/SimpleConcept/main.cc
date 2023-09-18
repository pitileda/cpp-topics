#include <concepts>
#include <iostream>
#include <type_traits>
#include <utility>

template <typename T>
concept NumberType = std::is_floating_point_v<T> || std::is_integral_v<T>;

/*template <NumberType Number>
auto sum(Number x, Number y) -> decltype(x + y) {
  return x + y;
};

template <NumberType T, NumberType U>
auto sum(T x, U y) -> decltype(x + y) {
  return x + y;
};
*/

template <NumberType Number>
auto sum(Number x) -> decltype(x) {
  return x;
};

template <NumberType T, NumberType... Args>
auto sum(T param1, Args... params) {
  return param1 + sum(params...);
};

int main(int argc, char const *argv[]) {
  auto a = sum(1, 2);
  auto b = sum(1.3, 2.4);
  auto c = sum(1, 2.3);
  auto d = sum(1, 2, 3.3, 4);
  std::cout << a << std::endl
            << b << std::endl
            << c << std::endl
            << d << std::endl;
  // no operand of the disjunction is satisfied
  // std::cout << sum("hi", "ho") << std::endl;
  return 0;
}
