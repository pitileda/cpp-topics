#include <iostream>
#include <ostream>

#include "square_lib/square.h"

int main(int argc, char const *argv[]) {
  std::cout << "Hello, Bazel!!!" << std::endl;
  std::cout << Square(16) << std::endl;
  return 0;
}