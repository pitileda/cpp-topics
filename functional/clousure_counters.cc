#include <functional>
#include <iostream>
#include <ostream>
#include <thread>

struct Func {
  int mutable n = 0;
  int operator()() const { return ++n; }
};

int incr(int& n) { return n++; }

std::function<int()> the_counter() {
  int n = 0;
  return [n]() mutable -> int { return n++; };
}

int main(int argc, char const* argv[]) {
  // auto n = 4;
  // auto counter = [n]() mutable { return ++n; };
  // auto count = [counter]() mutable { return counter(); };
  // std::cout << counter() << std::endl;
  // std::cout << counter() << std::endl;
  // std::cout << counter() << std::endl;
  // std::cout << count() << std::endl;
  // std::cout << count() << std::endl;
  // auto anothe_count = count;
  // count();
  // std::cout << "count(): " << count() << std::endl;
  // std::cout << "anothe_count(): " << anothe_count() << std::endl;
  // std::cout << n << std::endl;

  // []() { std::cout << "Hello"; }();

  Func f;
  f();
  std::cout << f() << '\n';

  auto counter = []() {
    int n = 0;
    return [x = std::move(n)]() mutable { return x++; };
  };
  auto count = counter();
  std::cout << count() << '\n';
  std::cout << count() << '\n';
  std::cout << count() << '\n';

  auto second = count;
  std::cout << "1st counter: " << count() << '\n';
  std::cout << "2nd counter: " << second() << '\n';
  std::cout << "2nd counter: " << second() << '\n';
  std::thread t1([&count]() {
    for (int i = 0; i < 10000; ++i) {
      count();
    }
  });
  t1.join();
  std::cout << "1st counter: " << count() << '\n';
  std::cout << "2nd counter: " << second() << '\n';

  auto third = the_counter();
  std::cout << "3rd counter: " << third() << '\n';
  std::cout << "3rd counter: " << third() << '\n';
  return 0;
}