#include <iostream>
#include <memory>
#include <vector>
class Shape {
 public:
  virtual int getArea() = 0;
  virtual ~Shape(){};
};

class Rectagle : public Shape {
 private:
  using width = int;
  using height = int;
  width w;
  height h;

 public:
  explicit Rectagle(width w, height h) : w(w), h(h) {}
  int getArea() override {
    std::cout << "Rec\n";
    return w * h;
  }
};

class Circle : public Shape {
 private:
  int radius;

 public:
  explicit Circle(int radius) : radius(radius) {}
  int getArea() override {
    std::cout << "Cir\n";
    return 3.14 * radius * radius;
  }
};

int main(int argc, char const* argv[]) {
  std::vector<std::unique_ptr<Shape>> shapes;
  shapes.push_back(std::make_unique<Rectagle>(12, 4));
  shapes.push_back(std::make_unique<Circle>(12));
  for (const auto& shape : shapes) {
    std::cout << shape->getArea() << std::endl;
  }
  return 0;
}