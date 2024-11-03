#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <semaphore>
#include <thread>

class H2O {
 private:
  std::atomic<int> h_count{2};       // Hydrogen count
  std::atomic<bool> o_count{false};  // Oxygen count
  std::mutex mtx;
  std::condition_variable cv;

 public:
  H2O() = default;

  void hydrogen(std::function<void()> releaseHydrogen) {
    --h_count;
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck,
            [this] { return h_count.load() == 0 && o_count.load() == true; });
    releaseHydrogen();
    h_count.store(2);
    cv.notify_all();
  }

  void oxygen(std::function<void()> releaseOxygen) {
    o_count = true;
    cv.notify_all();
    releaseOxygen();
    o_count = false;
  }
};

int main(int argc, char const *argv[]) {
  auto printH = [] { std::cout << 'H'; };
  auto printO = [] { std::cout << 'O'; };
  H2O water;

  std::thread h1(&H2O::hydrogen, &water, printH);
  std::thread o1(&H2O::oxygen, &water, printO);
  std::thread h2(&H2O::hydrogen, &water, printH);

  std::thread o2(&H2O::oxygen, &water, printO);
  std::thread h3(&H2O::hydrogen, &water, printH);
  std::thread h4(&H2O::hydrogen, &water, printH);

  h1.join();
  o1.join();
  h2.join();
  o2.join();
  h3.join();
  h4.join();

  return 0;
}