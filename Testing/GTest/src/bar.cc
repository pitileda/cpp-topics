#include "bar.h"

int Bar::Do(int a) {
  if (a == 0) {
    return 12;
  }
  if (Oops(a) > 10) {
    return 100;
  }

  return 200;
}

// LCOV_EXCL_START
int Bar ::Oops(int b) {
  std::cout << "Foo::Oops " << b << std::endl;
  return 42;
}
// LCOV_EXCL_STOP