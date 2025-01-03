#include <coroutine>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ostream>

template <typename T>
struct MyFuture {
  std::shared_ptr<T> value;
  MyFuture(std::shared_ptr<T> p) : value(p) {}
  ~MyFuture() {}
  T get() { return *value; }

  struct promise_type {
    std::shared_ptr<T> ptr = std::make_shared<T>();
    ~promise_type() {}
    MyFuture<T> get_return_object() { return ptr; }

    void return_value(T v) { *ptr = v; }

    std::suspend_never initial_suspend() { return {}; }

    std::suspend_never final_suspend() noexcept { return {}; }

    void unhandled_exception() { std::exit(1); }
  };
};

MyFuture<int> create_future() { co_return 2021; }

int main(int argc, char const *argv[]) {
  auto fut = create_future();
  std::cout << fut.get() << std::endl;
  return 0;
}