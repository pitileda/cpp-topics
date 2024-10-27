#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
using namespace std;

map<string, int> book{{"Shevchenko", 1845}, {"Hello", 2013}};

shared_timed_mutex book_mutex;

// writer
auto addToBook(const string& name, int year) ->void {
  lock_guard<shared_timed_mutex> write_guard(book_mutex);
  cout << "start update\n";
  this_thread::sleep_for(chrono::milliseconds(100));
  book[name] = year;
  cout << "end update\n";
}

// reader
auto print(const string& name) ->void {
  shared_lock<shared_timed_mutex> reader_lock(book_mutex);
  if (book.count(name) > 0) {
    cout << name << ": " << book[name] << endl;
  }
}

int main(int argc, char *argv[])
{
  thread reader_01([](){
    while (true) {
      if (book.count("Ihor") > 0) {
        print("Ihor");
        break;
      }
      this_thread::sleep_for(chrono::milliseconds(2000));
    }
  });
  thread reader_02([](){
    while (true) {
      if (book.count("Lena") > 0) {
        print("Lena");
        break;
      }
      this_thread::sleep_for(chrono::milliseconds(2000));
    }
  });
  thread reader1([](){print("Shevchenko");});
  thread reader2([](){print("Hello");});
  thread reader3([](){print("Oops");});
  thread writer1([](){addToBook("Ihor", 2024);});
  thread writer2([](){addToBook("Lena", 2024);});

  reader_01.join();
  reader_02.join();

  reader1.join();
  reader2.join();
  reader3.join();

  writer1.join();
  writer2.join();
  return 0;
}
