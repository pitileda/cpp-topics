#include <chrono>
#include <iostream>
#include <mutex>
#include <thread>
using namespace std;

struct Critical {
  mutex m;
};

void fun(Critical& a, Critical& b) {
  a.m.lock();
  cout << "1st mutex held\n";
  this_thread::sleep_for(chrono::milliseconds(5));
  b.m.lock();
  cout << "2nd mutex held\n";
  this_thread::sleep_for(chrono::milliseconds(5));
  a.m.unlock();
  b.m.unlock();
}

// fix with unique_lock and defer_lock
auto fun_fix_unique(Critical& a, Critical& b) -> void {
  unique_lock<mutex> a_lock(a.m, defer_lock);
  cout << "thread " << this_thread::get_id() << "got 1st mutex\n";
  this_thread::sleep_for(chrono::milliseconds(5));
  unique_lock<mutex> b_lock(b.m, defer_lock);
  cout << "thread " << this_thread::get_id() << "got 2nd mutex\n";
  this_thread::sleep_for(chrono::milliseconds(5));
  lock(a_lock, b_lock);
}

// fix with lock_guard and adopt_lock
auto fun_fix_guard(Critical& a, Critical& b) -> void {
  lock(a.m, b.m);
  lock_guard<mutex> a_lock(a.m, adopt_lock);
  cout << "thread " << this_thread::get_id() << "got 1st mutex\n";
  this_thread::sleep_for(chrono::milliseconds(5));
  lock_guard<mutex> b_lock(b.m, adopt_lock);
  cout << "thread " << this_thread::get_id() << "got 2nd mutex\n";
  this_thread::sleep_for(chrono::milliseconds(5));
}

int main(int argc, char *argv[])
{
  Critical c1;
  Critical c2;

  // deadlock
  //thread t(fun, ref(c1), ref(c2));
  // thread t2(fun, ref(c2), ref(c1));

  thread t(fun_fix_unique, ref(c1), ref(c2));
  thread t2(fun_fix_unique, ref(c2), ref(c1));

  thread t3(fun_fix_guard, ref(c1), ref(c2));
  thread t4(fun_fix_guard, ref(c2), ref(c1));

  t.join();
  t2.join();
  t3.join();
  t4.join();
  return 0;
}
