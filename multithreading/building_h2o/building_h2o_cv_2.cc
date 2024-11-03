#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

using namespace std;

class H2O {
  mutex excl;
  condition_variable cond;
  int hydro_out;
  int oxy_out;

 public:
  H2O() : hydro_out(0), oxy_out(0) {}

  void locked_try_rst() {
    if (hydro_out == 2 && oxy_out == 1) {
      hydro_out = 0;
      oxy_out = 0;
      cond.notify_all();
    }
  }

  void hydrogen(function<void()> releaseHydrogen) {
    unique_lock<mutex> lock(excl);
    cond.wait(lock, [this] { return hydro_out < 2; });
    ++hydro_out;
    releaseHydrogen();
    locked_try_rst();
  }

  void oxygen(function<void()> releaseOxygen) {
    unique_lock<mutex> lock(excl);
    cond.wait(lock, [this] { return oxy_out < 1; });
    ++oxy_out;
    releaseOxygen();
    locked_try_rst();
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