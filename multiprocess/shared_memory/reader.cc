#include <iostream>
#include <vector>
#include <cstring>
#include <memory>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

class Shape {
public:
    virtual ~Shape() = default;
    virtual void draw() const = 0;
    static std::unique_ptr<Shape> deserialize(const char*& data);
};

class Circle : public Shape {
public:
    float radius;
    Circle(float r = 0.0f) : radius(r) {}

    void draw() const override {
        std::cout << "Drawing Circle with radius " << radius << '\n';
    }
};

class Rectangle : public Shape {
public:
    float width, height;
    Rectangle(float w = 0.0f, float h = 0.0f) : width(w), height(h) {}

    void draw() const override {
        std::cout << "Drawing Rectangle with width " << width
                  << " and height " << height << '\n';
    }
};

std::unique_ptr<Shape> Shape::deserialize(const char*& data) {
    int type;
    std::memcpy(&type, data, sizeof(type));
    data += sizeof(type);

    if (type == 1) {
        float radius;
        std::memcpy(&radius, data, sizeof(radius));
        data += sizeof(radius);
        return std::make_unique<Circle>(radius);
    } else if (type == 2) {
        float width, height;
        std::memcpy(&width, data, sizeof(width));
        data += sizeof(width);
        std::memcpy(&height, data, sizeof(height));
        data += sizeof(height);
        return std::make_unique<Rectangle>(width, height);
    }

    throw std::runtime_error("Unknown type");
}

int main() {
    const char* shm_name = "/shape_shm";
    const size_t shm_size = 1024;

    int shm_fd = shm_open(shm_name, O_RDONLY, 0666);
    if (shm_fd == -1) {
        perror("shm_open");
        return 1;
    }

    void* ptr = mmap(nullptr, shm_size, PROT_READ, MAP_SHARED, shm_fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    const char* data = static_cast<const char*>(ptr);
    const char* end = data + shm_size;

    std::cout << "Reading shapes from shared memory:\n";

    while (data < end) {
        try {
            auto shape = Shape::deserialize(data);
            shape->draw();
        } catch (...) {
            break; // Ошибка или конец корректных данных
        }
    }

    munmap((void*)ptr, shm_size);
    close(shm_fd);
    shm_unlink(shm_name); // Удаляем shared memory

    return 0;
}
