#include <iostream>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

template <typename T>
class Property {
public:
  explicit Property(T Value) : value(std::move(Value)) {}
  explicit operator T() {
    return value;
  }
  Property& operator=(T v) {
    value = std::move(v);
    return *this;
  }
  T operator*(const T& other) {
    return value * other;
  }
  T operator+(const T& other) {
    return value + other;
  }
  friend std::ostream& operator<<(std::ostream& os, const Property<T>& prop) {
    return os << prop.value;
  }
private:
  T value;
};

struct Creature {
  Creature(const int &Power, const std::string &Name) : power(Power), name(Name) {}
  Property<int> power;
  Property<std::string> name;
friend std::ostream &operator<<(std::ostream &Os, const Creature &Obj) {
    return Os << "power: " << Obj.power << " name: " << Obj.name;
  }
};

class Modificator {
public:
  virtual ~Modificator() = default;
  explicit Modificator(Creature &Next) : creature(Next){}

  void add(std::unique_ptr<Modificator> Next) {
    if (next_modificator) {
      next_modificator->add(std::move(Next));
    } else {
      next_modificator = std::move(Next);
    }
  }

  virtual void handle() {
    if (next_modificator) {
      next_modificator->handle();
    }
  }
protected:
  Creature& creature;
private:
  std::unique_ptr<Modificator> next_modificator;
  size_t next = -1;
};

class Duplicator : public Modificator {
public:
  explicit Duplicator(Creature & Next) : Modificator(Next) {}
  void handle() override {
    creature.power = creature.power * 2;
    Modificator::handle();
  }
};

class Adder : public Modificator {
public:
  explicit Adder(Creature & Next) : Modificator(Next) {}
  void handle() override {
    creature.power = creature.power + 2;
    Modificator::handle();
  }
};

int main() {
  Creature creature(11, "Boo");
  Modificator root(creature);
  root.add(std::make_unique<Duplicator>(creature));
  root.add(std::make_unique<Adder>(creature));
  root.add(std::make_unique<Duplicator>(creature));
  root.add(std::make_unique<Adder>(creature));
  root.handle();
  std::cout << creature.name << creature.power <<std::endl;
  return 0;
}
