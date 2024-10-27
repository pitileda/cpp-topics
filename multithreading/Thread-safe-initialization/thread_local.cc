#include <iostream>
#include <mutex>
#include <string>
#include <thread>
using namespace std;

thread_local string s = "Hello ";
mutex coutMutex;

auto log(const string& name) -> void {
  s += name;
  lock_guard<mutex> lck(coutMutex);
  cout << s << endl;
}

int main(int argc, char *argv[])
{
  thread t1([](){log("Ihor");});
  thread t2([](){log("Lena");});

  t1.join();
  t2.join();
  return 0;
}
