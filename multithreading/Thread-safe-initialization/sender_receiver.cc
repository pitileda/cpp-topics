#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
using namespace std;

mutex mtx;
condition_variable cv;

map<string, int> book{{"Hello", 123}, {"Good", 12}};

auto do_the_work(pair<string, int> item) -> void {
  book[item.first] = item.second;
}

auto wait_for_updates(const string &s) -> void {
  cout << "Start to read for " << s << endl;
  unique_lock<mutex> lck(mtx);
  cv.wait(lck);
  if (book.count(s) > 0) {
    cout << book[s] << endl;
  }
}

auto update_book(pair<string, int> item) -> void {
  cout << "trying to update...\n";
  this_thread::sleep_for(chrono::milliseconds(1500));
  do_the_work(item);
  cv.notify_one();
}

int main(int argc, char *argv[]) {
  thread consumer(wait_for_updates, string("Win"));
  thread producer(update_book, make_pair<string, int>("Win", 1234));

  consumer.join();
  producer.join();
  return 0;
}
