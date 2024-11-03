#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

class FooBar {
 private:
  int n;
  mutex mtx;
  condition_variable cv;
  bool foo_turn = true;

 public:
  FooBar(int n) { this->n = n; }

  void foo(function<void()> printFoo) {
    for (int i = 0; i < n; i++) {
      unique_lock<mutex> lck(mtx);
      cv.wait(lck, [this] { return foo_turn; });
      // printFoo() outputs "foo". Do not change or remove this line.
      printFoo();
      foo_turn = false;
      cv.notify_all();
    }
  }

  void bar(function<void()> printBar) {
    for (int i = 0; i < n; i++) {
      unique_lock<mutex> lck(mtx);
      cv.wait(lck, [this] { return !foo_turn; });
      // printBar() outputs "bar". Do not change or remove this line.
      printBar();
      foo_turn = true;
      cv.notify_all();
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