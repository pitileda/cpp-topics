#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <vector>

class Shape {
 public:
  virtual ~Shape() = default;
  virtual void serialize(std::vector<char>& buffer) const = 0;
};

class Circle : public Shape {
 public:
  float radius;

  Circle(float r = 0.0f) : radius(r) {}

  void serialize(std::vector<char>& buffer) const override {
    int type = 1;
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&type),
                  reinterpret_cast<const char*>(&type) + sizeof(type));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&radius),
                  reinterpret_cast<const char*>(&radius) + sizeof(radius));
  }
};

class Rectangle : public Shape {
 public:
  float width, height;

  Rectangle(float w = 0.0f, float h = 0.0f) : width(w), height(h) {}

  void serialize(std::vector<char>& buffer) const override {
    int type = 2;
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&type),
                  reinterpret_cast<const char*>(&type) + sizeof(type));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&width),
                  reinterpret_cast<const char*>(&width) + sizeof(width));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&height),
                  reinterpret_cast<const char*>(&height) + sizeof(height));
  }
};

int main() {
  const char* shm_name = "/shape_shm";
  const size_t shm_size = 1024;

  // Создаём shared memory
  int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    perror("shm_open");
    return 1;
  }

  ftruncate(shm_fd, shm_size);

  void* ptr =
      mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (ptr == MAP_FAILED) {
    perror("mmap");
    return 1;
  }

  // Сериализуем объекты
  Circle c(5.0f);
  Rectangle r(10.0f, 20.0f);

  std::vector<char> buffer;
  c.serialize(buffer);
  r.serialize(buffer);

  std::memcpy(ptr, buffer.data(), buffer.size());

  std::cout << "Shapes written to shared memory.\n";
  std::cout << "Press Enter to exit (but keep shared memory).\n";
  std::cin.get();

  // Shared memory остаётся, unlink делает второй процесс
  munmap(ptr, shm_size);
  close(shm_fd);
  return 0;
}
