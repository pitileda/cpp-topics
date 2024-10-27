#include <iostream>
#include <thread>

using namespace std;

struct Double
{
  constexpr Double(double v): value_(v) {}
  constexpr double get() const {return value_;}

  private:
  double value_;
};

int main() {
  cout << "Hello World!" << endl;
  constexpr Double d(10.5);
  std::thread t1([&d](){d.get();});
  std::thread t2([&d](){d.get();});

  t1.join();
  t2.join();
  return 0;
}
