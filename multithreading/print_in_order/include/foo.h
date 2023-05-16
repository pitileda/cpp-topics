#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>

namespace cv {
class Foo {
 private:
  std::mutex mtx_;
  std::condition_variable cv;
  enum class Order { one, two, three } order_;

 public:
  Foo() { order_ = Order::one; }

  void first(std::function<void()> printFirst) {
    // printFirst() outputs "first". Do not change or remove this line.
    std::unique_lock<std::mutex> lk(mtx_);
    printFirst();
    order_ = Order::two;
    lk.unlock();
    cv.notify_all();
  }

  void second(std::function<void()> printSecond) {
    // printSecond() outputs "second". Do not change or remove this line.
    std::unique_lock<std::mutex> lk(mtx_);
    cv.wait(lk, [this]() { return order_ == Foo::Order::two; });
    printSecond();
    order_ = Order::three;
    lk.unlock();
    cv.notify_all();
  }

  void third(std::function<void()> printThird) {
    // printThird() outputs "third". Do not change or remove this line.
    std::unique_lock<std::mutex> lk(mtx_);
    cv.wait(lk, [this]() { return order_ == Foo::Order::three; });
    printThird();
  }
};
}  // namespace cv