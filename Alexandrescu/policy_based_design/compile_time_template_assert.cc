#include <iostream>
#include <ostream>

namespace simple {

template <bool>
struct CompileTimeError;

template <>
struct CompileTimeError<true> {};

#define STATIC_CHECK_SIMPLE(expr) (CompileTimeError<(expr) != 0>())

template <class To, class From>
To safe_reinterpret_cast(From from) {
  STATIC_CHECK_SIMPLE(sizeof(From) <= sizeof(To));
  return reinterpret_cast<To>(from);
}

}  // namespace simple

namespace complicated {

template <bool>
struct CompileTimeChecker {
  static void check() {}
};
template <>
struct CompileTimeChecker<false> {};
#define STATIC_CHECK_COMPLEX(expr, msg)       \
  {                                           \
    class ERROR_##msg {};                     \
    CompileTimeChecker<(expr) != 0>::check(); \
  }

template <class To, class From>
To safe_reinterpret_cast(From from) {
  STATIC_CHECK_COMPLEX(sizeof(From) <= sizeof(To), Destination_Type_Too_Narrow);
  return reinterpret_cast<To>(from);
}

}  // namespace complicated

int main() {
  void* somePointer = new long;
  // char c = simple::safe_reinterpret_cast<char>(somePointer);
  long l = simple::safe_reinterpret_cast<long>(somePointer);
  // char cc = complicated::safe_reinterpret_cast<char>(somePointer);
  long ll = complicated::safe_reinterpret_cast<long>(somePointer);
  std::cout << "somePointer addr: " << &somePointer << ". " << std::endl;
  std::cout << "somePointer value: " << *(static_cast<long*>(somePointer))
            << ". " << std::endl;
  std::cout << "l addr: " << &l << ". " << std::endl;
  std::cout << "l value: " << l << ". " << std::endl;
  std::cout << "ll addr: " << &ll << ". " << std::endl;
  std::cout << "ll value: " << ll << ". " << std::endl;
  return 0;
}