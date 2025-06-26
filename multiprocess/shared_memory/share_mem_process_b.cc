#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

int main() {
  const char* name = "/my_shared_memory";
  const size_t SIZE = 4096;

  // Открываем существующую разделяемую память
  int shm_fd = shm_open(name, O_RDONLY, 0666);
  if (shm_fd == -1) {
    perror("shm_open");
    return 1;
  }

  void* ptr = mmap(0, SIZE, PROT_READ, MAP_SHARED, shm_fd, 0);

  // Читаем данные
  std::cout << "Message from shared memory: " << static_cast<char*>(ptr)
            << '\n';

  return 0;
}
