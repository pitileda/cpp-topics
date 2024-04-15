#include "static_containers.h"
#include <iostream>

int main(int argc, char const *argv[]) {
  sc::Queue<int, 5> q;
  q.push(1);
  q.push(2);
  q.push(3);
  q.push(4);
  q.push(5);
  q.pop();
  q.push(6);
  if (q.push(100) == Code::ErrorSize) {
    std::cout << "Array is Full" << std::endl;
  }
  auto [val, err] = q.get(0);
  if (err == Code::Success) {
    std::cout << val << std::endl;
  }
  auto result = q.get(1);
  val = result.first;
  err = result.second;
  if (err == Code::Success) {
    std::cout << val << std::endl;
  }

  sc::Stack<int, 6> s;
  s.push(12);
  result = s.get(0);
  val = result.first;
  err = result.second;
  if (err == Code::Success) {
    std::cout << val << std::endl;
  }
  s.push(13);
  s.pop();
  s.pop();
  s.pop();

  result = s.get(0);
  val = result.first;
  err = result.second;
  if (err == Code::Success) {
    std::cout << val << std::endl;
  }

  sc::StaticArray return 0;
}
