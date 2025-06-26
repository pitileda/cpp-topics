// c++20 concepts

#include <iostream>
#include <type_traits>

template <typename T>
constexpr bool IsPointer = std::is_pointer_v<T>;

template <typename T>
  requires IsPointer<T>
void func(T value) {
  std::cout << "Pointer version\n";
  std::cout << value << std::endl;
}

template <typename T>
  requires(!IsPointer<T>)
void func(T value) {
  std::cout << "General version\n";
  std::cout << value << std::endl;
}

template <typename T>
concept IntegralValue = std::is_integral_v<T>;

template <typename T>
  requires IntegralValue<T>
void foo(T value) {
  std::cout << "Integral version\n";
  std::cout << value << std::endl;
}

template <IntegralValue T>
T bar(const T& arg) {
  std::cout << "bar is integral\n";
  std::cout << arg << std::endl;
  return arg;
}

int main() {
  // dummy(0.1);
  // dummy(100);
  int x = 12;
  func(x);
  func(&x);
  func(12);
  foo(x);
  bar(true);
  bar('c');
  // bar(std::string("Hello")); // compile error
  return 0;
}