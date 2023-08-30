#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <future>
#include <iostream>
#include <string>
#include <thread>

std::string longOperation() {
  std::this_thread::sleep_for(std::chrono::seconds(5));
  return "From child!!!";
}

void handleTimeout(int signum) {
  std::cout << "Timeout reached. Terminating child process." << std::endl;
  _exit(1);
}

int main(int argc, char const* argv[]) {
  // std::thread t([]() { std::cout << "Hello from Thread" << '\n'; });
  // std::thread t(longOperation);
  // t.join();
  int pipefd[2];
  if (pipe(pipefd) == -1) {
    perror("pipe");
    return 1;
  }

  pid_t pid = fork();
  if (pid == 0) {
    signal(SIGALRM, handleTimeout);
    alarm(2);
    std::string result = longOperation();
    close(pipefd[0]);
    write(pipefd[1], result.c_str(), result.size() + 1);
    close(pipefd[1]);
    // close(pipefd[0]);
    alarm(0);
    _exit(0);
  } else if (pid > 0) {
    waitpid(pid, nullptr, 0);
    char buffer[256];
    close(pipefd[1]);
    ssize_t bytesRead = read(pipefd[0], buffer, sizeof(buffer));
    close(pipefd[0]);
    std::string result;
    if (bytesRead > 0) {
      result = buffer;
      std::cout << "result is: " << result << '\n';
    } else {
      std::cout << "Fail" << '\n';
    }
    close(pipefd[0]);
    close(pipefd[1]);
  } else {
    return 2;
  }
  std::cout << "Hello!" << '\n';
  std::this_thread::sleep_for(std::chrono::seconds(2));
  std::cout << "I'm here" << '\n';
  return 0;
}
