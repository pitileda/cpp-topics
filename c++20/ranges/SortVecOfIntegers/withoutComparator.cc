#include <algorithm>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

struct Task {
  std::string desc;
  unsigned int priority{0};
};

int main(int argc, char const* argv[]) {
  std::vector<Task> tasks{{"clean up my room", 10},
                          {"finish homework", 5},
                          {"test a car", 8},
                          {"buy new monitor", 12}};
  std::ranges::sort(tasks, std::ranges::greater{}, &Task::priority);
  auto print = [](const Task& task) {
    std::cout << task.desc << ", priority: " << task.priority << '\n';
  };
  std::ranges::for_each(tasks, print);
  return 0;
}
