#include <cstdint>
#include <iostream>
#include <utility>
class Foo {
 private:
  static uint64_t counter;

 public:
  Foo() noexcept { counter++; }
  Foo(const Foo&) noexcept { counter++; }
  Foo(const Foo&&) noexcept {
    // default constructed move ctor will use copy ctor
    // Foo mf = std::move(cf);
    // so, Foo(const Foo& cf) will be called
  }
  Foo& operator=(const Foo&) noexcept { return *this; }
  Foo& operator=(const Foo&&) noexcept {
    counter--;
    return *this;
  }
  ~Foo() { counter--; }

  static uint64_t count() { return counter; }
};

uint64_t Foo::counter = 0;

int main(int argc, char const* argv[]) {
  Foo f1;  // 1 - f1
  { Foo f2; }
  Foo f3 = f1;             // 2 - f1, f3
  Foo f4 = std::move(f3);  // 2 - f1, f4
  Foo f5;                  // 3 - f1, f4, f5
  Foo f6;                  // 4 - f1, f4, f5, f6
  f5 = f1;                 // 4 - f1, f4, f5, f6
  f6 = std::move(f5);      // 3 - f1, f4, f6
  std::cout << Foo::count();
  return 0;
}