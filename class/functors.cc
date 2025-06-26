#include <functional>
#include <iostream>

struct Greater {
  bool operator()(int a, int b) const { return a > b; }
};

struct Less {
  bool operator()(int a, int b) const { return a < b; }
};

int main() {
  int a = 5, b = 10;
  bool useGreater = true;

  std::function<bool(int, int)> comp;
  useGreater ? comp = Greater() : comp = Less();

  if (comp(a, b)) {
    std::cout << "Condition met\n";
  } else {
    std::cout << "Condition not met\n";
  }
}
