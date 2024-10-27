#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <thread>

using namespace std;

once_flag flag;

void do_only_once() {
  call_once(flag, [](){cout << "should be printed only once\n";});
};

class Single {
 private:
  Single() = default;
  static once_flag flag;
  static unique_ptr<Single> instance;
 public:
  Single(const Single&) = delete;
  Single& operator=(const Single&) = delete;

  static Single& get() {
    call_once(flag, [](){
      instance = unique_ptr<Single>(new Single);
    });
    return *instance;
  }
};

once_flag Single::flag;
unique_ptr<Single> Single::instance = nullptr;

int main(int argc, char *argv[])
{
  thread t1(do_only_once);
  thread t2(do_only_once);
  thread t3(do_only_once);
  thread t4(do_only_once);

  t1.join();
  t2.join();
  t3.join();
  t4.detach();

  cout << "Sinlgeton get: " << &(Single::get()) << endl;
  cout << "Sinlgeton get: " << &(Single::get()) << endl;
  return 0;
}
