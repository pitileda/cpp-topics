#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

using namespace std;

struct Data {
  int id;
  double value;
};

mutex mtx;
condition_variable cv;
bool done = false;

Data pop_data(queue<Data> &q) {
  unique_lock<mutex> lck(mtx);
  cv.wait(lck, [&] { return !q.empty() || done; });
  Data res = q.front();
  q.pop();
  return res;
}

void push_data(queue<Data> &q, const Data &data) {
  lock_guard<mutex> guard(mtx);
  q.push(data);
  cv.notify_all();
}

int main() {
  cout << "Hello World!" << endl;
  queue<Data> q;
  thread w1([&]() {
    for (int iteration = 0; iteration < 10000; ++iteration) {
      push_data(q, {iteration, static_cast<double>(iteration)});
      cout << "w1q";
      // this_thread::sleep_for(chrono::milliseconds(1));
    }
    cout << "all iterations done\n";
  });

  thread w2([&]() {
    for (int iteration = 0; iteration < 1000; ++iteration) {
      push_data(q, {iteration, static_cast<double>(iteration)});
      cout << "w2t";
      // this_thread::sleep_for(chrono::milliseconds(1));
    }
    cout << "w2 all iterations done\n";
  });

  auto reader = [&](int n) {
    for (int iteration = 0; iteration < 6000; ++iteration) {
      pop_data(q);
      cout << n;
      // this_thread::sleep_for(chrono::milliseconds(2));
    }
  };

  thread r1(reader, 1);
  thread r2(reader, 2);
  thread r3(reader, 3);
  thread r4(reader, 4);

  w1.join();
  w2.join();

  {
    lock_guard<mutex> guard(mtx);
    done = true;
    cv.notify_all(); // notify all consumers to check the `done` flag
  }

  r1.join();
  r2.join();
  r3.join();
  r4.join();
  return 0;
}
