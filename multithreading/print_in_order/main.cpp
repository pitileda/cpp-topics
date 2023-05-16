#include <iostream>
#include <thread>

#include "bar.h"
#include "foo.h"
#include "zoo.h"

#include <vector>

int main(int, char**) {
  // std::cout << "Hello, world!\n";

  // using cv::Foo;
  // Foo foo;
  // std::thread t1(&Foo::third, &foo, []() { std::cout << "3rd"; });
  // std::thread t2(&Foo::second, &foo, []() { std::cout << "2nd"; });
  // std::thread t3(&Foo::first, &foo, []() { std::cout << "1st"; });

  // t1.join();
  // t2.join();
  // t3.join();

  // std::cout << std::endl;

  // using prms::Bar;
  // Bar bar;
  // t1 = std::thread(&Bar::third, &bar, []() { std::cout << "bar3"; });
  // t2 = std::thread(&Bar::second, &bar, []() { std::cout << "bar2"; });
  // t3 = std::thread(&Bar::first, &bar, []() { std::cout << "bar1"; });

  // t1.join();
  // t2.join();
  // t3.join();

  using sdl::Zoo;
  Zoo zoo;
  std::thread t1 = std::thread(&Zoo::threadFuncOne, &zoo);
  std::thread t2 = std::thread([&zoo](){
    std::chrono::milliseconds interval(100);
    std::this_thread::sleep_for(interval);
    zoo.threadFuncTwo();
  });

  t1.join();
  t2.join();

  std::cout << std::endl;
}
