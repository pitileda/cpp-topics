#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

class H2O {
 public:
  int hydrogenCount, oxygenCount;
  mutex m;
  condition_variable cv;
  bool start = false;
  H2O() {
    hydrogenCount = 0;
    oxygenCount = 0;
  }

  void hydrogen(function<void()> releaseHydrogen) {
    // Check if Oxygen atom is available by oxygenCount == 1
    unique_lock<mutex> lck(m);
    cv.wait(lck, [this]() { return oxygenCount == 1; });
    hydrogenCount += 1;
    releaseHydrogen();
    if (hydrogenCount == 2 and oxygenCount == 1) {
      hydrogenCount = 0;
      oxygenCount = 0;
      start = false;
    }
    cv.notify_all();
  }

  void oxygen(function<void()> releaseOxygen) {
    // Check if there are 2 Hydrogen atoms avaiable by hydrogenCount == 2
    unique_lock<mutex> lck(m);
    cv.wait(lck, [this]() { return start == false || hydrogenCount == 2; });
    start = true;
    oxygenCount += 1;
    releaseOxygen();
    if (hydrogenCount == 2 and oxygenCount == 1) {
      hydrogenCount = 0;
      oxygenCount = 0;
    }
    cv.notify_all();
  }
};

int main(int argc, char const *argv[]) {
  auto printH = [] { std::cout << 'H'; };
  auto printO = [] { std::cout << 'O'; };
  H2O water;
  std::thread h1(&H2O::hydrogen, &water, printH);
  std::thread h2(&H2O::hydrogen, &water, printH);
  std::thread h3(&H2O::hydrogen, &water, printH);
  std::thread h4(&H2O::hydrogen, &water, printH);
  std::thread o1(&H2O::oxygen, &water, printO);
  std::thread o2(&H2O::oxygen, &water, printO);

  h1.join();
  h2.join();
  h3.join();
  h4.join();

  o1.join();
  o2.join();
  return 0;
}