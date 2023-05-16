#pragma once

#include <functional>
#include <future>

namespace prms {
class Bar {
 private:
  std::promise<void> one_, two_;

 public:
  Bar() {}

  void first(std::function<void()> printFirst) {
    // printFirst() outputs "first". Do not change or remove this line.
    printFirst();
    one_.set_value();
  }

  void second(std::function<void()> printSecond) {
    // printSecond() outputs "second". Do not change or remove this line.
    one_.get_future().get();
    printSecond();
    two_.set_value();
  }

  void third(std::function<void()> printThird) {
    // printThird() outputs "third". Do not change or remove this line.
    two_.get_future().get();
    printThird();
  }
};
}  // namespace prms