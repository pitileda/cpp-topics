#include <iostream>
#include <memory>
#include <vector>

class Shape {
 public:
  virtual double getArea() const = 0;
  virtual ~Shape() {};
};

class Rectagle : public Shape {
 private:
  using width = int;
  using height = int;
  width w;
  height h;

 public:
  explicit Rectagle(width w, height h) : w(w), h(h) {}
  double getArea() const override {
    std::cout << "Rec\n";
    return w * h;
  }
};

class Circle : public Shape {
 private:
  double radius;

 public:
  explicit Circle(double radius) : radius(radius) {}
  double getArea() const override {
    std::cout << "Cir\n";
    return 3.14 * radius * radius;
  }
};

int main() {
  using ShapePtr = std::unique_ptr<Shape>;
  std::vector<ShapePtr> shapes;
  shapes.push_back(std::make_unique<Rectagle>(12, 4));
  shapes.push_back(std::make_unique<Circle>(12));
  for (const auto& shape : shapes) {
    std::cout << shape->getArea() << std::endl;
  }

  std::cout << "Move to references\n";

  // the same with references
  using ShapeRef = std::reference_wrapper<Shape>;
  std::vector<ShapeRef> shapes_ref;
  Rectagle rect(12, 4);
  Circle circle(12);
  shapes_ref.push_back(rect);
  shapes_ref.push_back(circle);
  for (const auto& shape : shapes_ref) {
    std::cout << shape.get().getArea() << std::endl;
  }
  return 0;
}