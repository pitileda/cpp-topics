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
// condition_variable cv;
condition_variable cv_consumer;
condition_variable cv_producer;
const size_t max_size = 10;

Data pop_data(queue<Data> &q) {
  unique_lock<mutex> lck(mtx);
  cv_consumer.wait_for(lck, chrono::seconds(5), [&] {
    return !q.empty();
  }); // wait if queue is empty and producers are not done
  if (q.empty())
    return {-1, -1}; // sentinel value to signal no data if done
  Data res = q.front();
  q.pop();
  cv_producer.notify_all();
  return res;
}

void push_data(queue<Data> &q, const Data &data) {
  unique_lock<mutex> guard(mtx);
  cv_producer.wait(guard, [&] { return q.size() < max_size; });
  q.push(data);
  cv_consumer.notify_all();
}

int main() {
  cout << "Hello World!" << endl;
  queue<Data> q;

  // Producer threads
  thread w1([&]() {
    for (int iteration = 0; iteration < 100; ++iteration) {
      push_data(q, {iteration, static_cast<double>(iteration)});
      cout << "w1q";
      // this_thread::sleep_for(chrono::milliseconds(3));
    }
    cout << "all iterations done\n";
  });

  thread w2([&]() {
    for (int iteration = 0; iteration < 100; ++iteration) {
      push_data(q, {iteration, static_cast<double>(iteration)});
      cout << "w2t";
      // this_thread::sleep_for(chrono::milliseconds(2));
    }
    cout << "w2 all iterations done\n";
  });

  // Consumer function
  auto reader = [&](int n) {
    while (true) {
      Data data = pop_data(q);
      if (data.id == -1)
        break; // stop if sentinel value is returned
      cout << "Consumer " << n << " processed: " << data.id << ", "
           << data.value << endl;
      // this_thread::sleep_for(chrono::milliseconds(1));
    }
  };

  // Consumer threads
  thread r1(reader, 1);
  thread r2(reader, 2);
  thread r3(reader, 3);
  thread r4(reader, 4);

  // Wait for producers to finish
  w1.join();
  w2.join();

  // Wait for consumers to finish
  r1.join();
  r2.join();
  r3.join();
  r4.join();

  return 0;
}
