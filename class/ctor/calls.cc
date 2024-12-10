#include <iostream>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

class T {
 private:
  std::string s;

 public:
  T() { std::cout << "T()\n"; }
  T(const T& other) { std::cout << "T(&)\n"; }
  T(T&& other) {
    s = std::move(other.s);
    std::cout << "T(&&)\n";
  }
  std::string get() const { return s; }
};

T func() {
  T t1 = T();  // T()
  return t1;   // RVO is used, no call to T(&&)
}

template <typename T>
void foo(T&& other) {
  if (std::is_lvalue_reference_v<T>) {
    std::cout << "called T&" << std::endl;
  } else {
    std::cout << "called T&&" << std::endl;
  }
}

int main(int argc, char const* argv[]) {
  T t = func();        // T() from func, line 22
  T t2 = t;            // T(&)
  foo(t);              // called T(&)
  foo(T());            // T(), called T(&&)
  foo(std::move(t2));  // called T(&&)
  return 0;
}