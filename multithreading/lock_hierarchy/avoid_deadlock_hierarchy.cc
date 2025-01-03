#include <exception>
#include <iostream>
#include <mutex>
#include <thread>

#include "hierarchical_mutex.h"

thread_local unsigned long hierarchical_mutex::this_thread_value(ULONG_MAX);

hierarchical_mutex high(1000);
hierarchical_mutex low(500);
hierarchical_mutex other(100);

int do_low_level() {
  std::cout << "do_low_level\n";
  return 12;
}

int low_level_func() {
  std::lock_guard<hierarchical_mutex> lck(low);
  return do_low_level();
}

void high_level_staff(int low_value) { std::cout << "high_level_staff\n"; }

void high_level_func() {
  std::lock_guard<hierarchical_mutex> lck(high);
  std::cout << "high_level_func\n";
  high_level_staff(low_level_func());
}

void thread_a() { high_level_func(); }

void do_other() { std::cout << "do_other\n"; }

void other_func() {
  high_level_func();
  do_other();
}

void thread_b() {
  std::lock_guard<hierarchical_mutex> lck(other);
  other_func();
}

int main(int argc, char const *argv[]) {
  std::thread t1([] {
    try {
      thread_a();
    } catch (std::exception e) {
      std::cout << e.what();
    }
  });
  std::thread t2([] {
    try {
      thread_b();
    } catch (std::exception e) {
      std::cout << e.what();
    }
  });

  t1.join();
  t2.join();
  return 0;
}