// different return type based on overloading

#include <iostream>
#include <type_traits>

// this would be void func(T)
// where T should be pointer
template <typename T>
typename std::enable_if<std::is_pointer<T>::value>::type func(const T& value) {
  std::cout << "Pointer version\n";
}

// this would be void func(T)
// where T should be not a pointer
template <typename T>
typename std::enable_if<!std::is_pointer<T>::value>::type func(const T& value) {
  std::cout << "Regular version\n";
}

// dummy function is ambigious
// template <typename T>
// int dummy(T value) {
//   return 14;
// }

// template <typename T>
// double dummy(T value) {
//   return 3.14;
// }

// dummy fun Solution
template <typename T>
typename std::enable_if<std::is_floating_point_v<T>, double>::type dummy(
    T value) {
  std::cout << "double version is callled\n";
  return 3.14;
}

template <typename T>
typename std::enable_if<std::is_integral_v<T>, int>::type dummy(T value) {
  std::cout << "int version is callled\n";
  return 12;
}

int main() {
  dummy(0.1);
  dummy(100);
  int x = 12;
  func(x);
  func(&x);
  func(12);
  return 0;
}