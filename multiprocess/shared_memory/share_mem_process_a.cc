#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

int main() {
  const char* name = "/my_shared_memory";
  const size_t SIZE = 4096;

  // Создаем разделяемую память
  int shm_fd = shm_open(name, O_CREAT | O_RDWR, 0666);
  ftruncate(shm_fd, SIZE);

  void* ptr = mmap(0, SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);

  const char* message = "Hello from process 1!";
  std::memcpy(ptr, message, strlen(message) + 1);

  std::cout << "Message written to shared memory: " << message << '\n';
  std::cout << "Press Enter to exit...\n";
  std::cin.get();

  shm_unlink(name);  // Удаляем сегмент памяти после завершения
  return 0;
}
