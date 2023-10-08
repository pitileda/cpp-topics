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

int Bar ::Oops(int b) {
  std::cout << "Foo::Oops " << b << std::endl;
  return 42;
}