#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

void sigHandler(int sig);

std::promise<void> promise;
std::future<void> future = promise.get_future();

std::mutex mutex;
std::condition_variable cv;

int shared_fd = -1;
void* longOperation(void* arg) {
  // signal(SIGUSR1, sigHandler);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  int fd = 123;
  {
    std::lock_guard<std::mutex> lock(mutex);
    shared_fd = fd;
  }
  cv.notify_one();
  std::cout << "Success\n";
  promise.set_value();
  return NULL;
}

void sigHandler(int sig) {}

void runRepeatedly() {
  signal(SIGUSR1, sigHandler);
  pthread_t thread;
  pthread_create(&thread, NULL, longOperation, 0);
  std::chrono::seconds duration(3);
  if (future.wait_for(duration) == std::future_status::timeout) {
    pthread_kill(thread, SIGUSR1);
    std::cout << "operation takes too long" << '\n';
  } else {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, []() { return shared_fd != -1; });
    int fd = shared_fd;
    std::cout << "Received fd: " << fd << '\n';
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
  std::cout << "I'm here" << '\n';
  return 0;
}
