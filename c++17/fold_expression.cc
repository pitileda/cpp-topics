#include <iostream>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

// simple usage
template <typename... Args>
auto sum(Args... args) {
  return (args + ... + 0);
}

template <typename... Args>
auto product(Args... args) {
  return (args * ... * 1);
}

template <typename First, typename... Rest>
std::optional<double> divide(First first, Rest... rest) {
  bool isZero = false;

  double result = first;

  auto step = [&](auto x) {
    if (static_cast<double>(x) == 0.0) {
      isZero = true;
      return;
    }
    result /= x;
  };

  (step(rest), ...);
  if (isZero) {
    return std::nullopt;
  }
  return std::optional<double>(result);
}

// with cout and forwarding references
template <typename... Args>
void _printf(Args&&... args) {
  (std::cout << ... << std::forward<Args>(args)) << '\n';
}

// using comma operator
template <typename T, typename... Args>
void fillVec(std::vector<T>& v, Args&&... args) {
  (v.push_back(args), ...);
}

// with cout and comma operator to have spaces
template <typename... Args>
void _printfs(Args&&... args) {
  const char sep = ' ';
  ((std::cout << std::forward<Args>(args) << sep), ...);
  std::cout << std::endl;
}

int main() {
  _printf(12, 'c', 12.4);
  _printfs(12, 'c', 12.4);
  std::vector<char> cv;
  fillVec(cv, 'c', 'b', 'f', 'g');
  _printf(cv[0], cv[1], cv[2], cv[3]);
  _printfs(product(1, 3, 5, 8), product(1, 3, 0), product(1, 2, 3, 4));
  _printfs(divide(22, 10, 2.0).value());
  _printfs(divide(10).value());
  auto result = divide(12, 6, 0, 8);
  if (result == std::nullopt) {
    std::cout << "division by zero\n";
  }
  return sum(1, 2, 4, 5);
}