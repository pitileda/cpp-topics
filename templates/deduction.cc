#include <iostream>
int main(int argc, char const* argv[]) {
  int x = 12;
  const int& cx = x;

  const int* px = &x;
  const int* const cpx = px;

  std::cout << ++x << cx << *px << *cpx;
  return 0;
}