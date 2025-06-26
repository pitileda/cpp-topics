#include <iostream>
#include <string>

template <typename T> class Property {
public:
  T value;
  Property(const T &value) { *this = value; }
  T operator=(T v) {
    value = v;
    return value;
  }
  operator T() { return value; }
};

struct POD {
  Property<int> strength{10};
  Property<std::string> name{"Dummy"};
};

int main() {
  POD pod;
  pod.strength = 11;
  pod.name = std::string("Ihor");
  std::cout << (int)pod.strength << (std::string)pod.name << std::endl;
  return 0;
}
