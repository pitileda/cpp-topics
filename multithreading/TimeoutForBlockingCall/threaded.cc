#include <pthread.h>
#include <signal.h>

#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>

void sigHandler(int sig);

std::promise<int> promise;
std::future<int> future = promise.get_future();

int shared_fd = -1;
void* longOperation(void* arg) {
  std::string* s = static_cast<std::string*>(arg);
  // signal(SIGUSR1, sigHandler);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  int fd = 123;
  std::cout << *s << '\n';
  promise.set_value(fd);
  return NULL;
}

void sigHandler(int sig) {}

void runRepeatedly() {
  signal(SIGUSR1, sigHandler);
  pthread_t thread;
  std::string s{"Argument"};
  pthread_create(&thread, NULL, longOperation, &s);
  std::chrono::milliseconds duration(5000);

  auto status = future.wait_for(duration);
  if (status == std::future_status::timeout) {
    pthread_kill(thread, SIGUSR1);
    std::cout << "operation takes too long" << '\n';
    return;
  }
  if (status == std::future_status::ready) {
    std::cout << "operation succeded" << '\n';
    return;
  }
}

int main(int argc, char const* argv[]) {
  std::thread t(runRepeatedly);
  t.join();
  struct sigaction sa;
  sa.sa_handler = SIG_IGN;  // Ignore the signal in the main thread
  sa.sa_flags = 0;
  sigaction(SIGUSR1, &sa, NULL);

  // pthread_t thread;
  // pthread_create(&thread, NULL, longOperation, 0);

  // pthread_kill(thread, SIGUSR1);
  if (future.wait_for(std::chrono::seconds(1)) == std::future_status::ready) {
    std::cout << future.get() << std::endl;
  }
  std::cout << "I'm here" << '\n';
  return 0;
}
