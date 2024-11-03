
#include <barrier>
#include <functional>
#include <iostream>
#include <semaphore>
#include <thread>
class H2O {
 private:
  std::counting_semaphore<2> h_sem{2};
  std::binary_semaphore o_sem{1};
  std::barrier<> molecule{3};

 public:
  H2O() = default;
  ~H2O() = default;

  void hydrogen(std::function<void()> releaseHydrogen) {
    h_sem.acquire();
    molecule.arrive_and_wait();
    releaseHydrogen();
    h_sem.release();
  }

  void oxygen(std::function<void()> releaseOxygen) {
    o_sem.acquire();
    molecule.arrive_and_wait();
    releaseOxygen();
    o_sem.release();
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