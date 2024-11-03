#include <future>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>

auto fun(const std::string &s) -> std::string { return "Hello: " + s; }

class FunOb {
public:
  auto operator()(const std::string &s) -> std::string {
    return "Object: " + s;
  }
};

auto l_fun = [](const std::string &s) -> std::string { return "Lambda: " + s; };

int main(int argc, char *argv[]) {
  auto fut_fun = std::async(fun, "Ihor");
  auto fut_obj = std::async(FunOb(), "Lena");
  auto fut_lmd = std::async(l_fun, "Yahoo!");

  std::cout << fut_fun.get() << std::endl;
  std::cout << fut_obj.get() << std::endl;
  std::cout << fut_lmd.get() << std::endl;
  return 0;
}
