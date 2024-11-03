#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <semaphore>
#include <thread>

using namespace std;

class FooBar {
 private:
  int n;
  mutex mtx;
  condition_variable cv;
  binary_semaphore f{1}, b{0};

 public:
  FooBar(int n) { this->n = n; }

  void foo(function<void()> printFoo) {
    for (int i = 0; i < n; i++) {
      f.acquire();
      // printFoo() outputs "foo". Do not change or remove this line.
      printFoo();
      b.release();
    }
  }

  void bar(function<void()> printBar) {
    for (int i = 0; i < n; i++) {
      b.acquire();
      // printBar() outputs "bar". Do not change or remove this line.
      printBar();
      f.release();
    }
  }
};

int main(int argc, char const *argv[]) {
  FooBar fb{10};
  auto print_foo = [] { cout << "foo"; };
  auto print_bar = [] { cout << "bar"; };
  thread t1(&FooBar::bar, &fb, print_bar);
  thread t2(&FooBar::foo, &fb, print_foo);

  t1.join();
  t2.join();
  return 0;
}