#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

std::condition_variable cv;
std::mutex mtx;
bool odd_turn = false;

void print_odd() {
  for (size_t i = 1; i < 100; i += 2) {
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck, [] { return odd_turn == true; });
    printf("%zu ", i);
    odd_turn = false;
    // lck.unlock();
    cv.notify_one();
  }
}

void print_even() {
  for (size_t i = 0; i < 101; i += 2) {
    std::unique_lock<std::mutex> lck(mtx);
    cv.wait(lck, [] { return odd_turn == false; });
    printf("%zu ", i);
    odd_turn = true;
    // lck.unlock();
    cv.notify_one();
  }
}

int main() {
  std::thread t_odd{print_odd};
  std::thread t_even{print_even};

  t_odd.join();
  t_even.join();
  return 0;
}
