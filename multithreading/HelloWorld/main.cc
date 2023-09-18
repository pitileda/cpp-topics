#include <chrono>
#include <iostream>
#include <thread>

void hello() { std::cout << "Hello from different thread!\n"; }

class BackgroundTask {
public:
  void operator()() { std::cout << "Oops!\n"; }
};

class JoinThread {
  std::thread &thread_;

public:
  explicit JoinThread(std::thread &thread) : thread_(thread) {}
  JoinThread(const std::thread &) = delete;
  JoinThread &operator=(const std::thread &) = delete;

  ~JoinThread() {
    if (thread_.joinable()) {
      thread_.join();
    }
  }
};

int main(int argc, char const *argv[]) {
  std::thread t(hello);
  BackgroundTask b;
  // std::thread bt{BackgroundTask()}; // works
  // std::thread bt(BackgroundTask()); // doesn't work - it is a fun declaration
  // std::thread bt( (BackgroundTask()) ); // works
  auto l = []() {
    BackgroundTask b;
    b();
  };
  std::thread bt(BackgroundTask{});
  std::thread lt(l);
  t.join();
  std::thread tt(hello);
  JoinThread jt(tt); // will call join() in destructor
  bt.join();
  // bt.join();  // thread obj is no longer associated with any thread, join()
  //  could be called only once
  lt.detach();
  return 0;
}
