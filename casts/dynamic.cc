#include <iostream>
#include <memory>

// Downcasting from Base to Derived

class Base {
 public:
  virtual void foo() { std::cout << "Foo"; }
};

class Derived : public Base {
 public:
  void bar() { std::cout << "BAR\n"; }
};

class Fake {
 public:
  void zoo() { std::cout << "ZOO\n"; }
};

// Checking obj types at runtime

class Interface {
 public:
  virtual void exec() = 0;
};

class Executor1 : public Interface {
 public:
  void exec() override { std::cout << "Executor1\n"; }
};

class Executor2 : public Interface {
 public:
  void exec() override { std::cout << "Executor2\n"; }
  void extra() {
    exec();
    std::cout << "Executor2 extra action\n";
  }
};

// with help of dynamic_cast next fun can be one for
// all derived types of Interface
// to avoid this numeric if-conditions Visitor pattern can be used
void execute(Interface* obj) {
  Executor1* ex1 = dynamic_cast<Executor1*>(obj);
  if (ex1) {
    ex1->exec();
  }
  Executor2* ex2 = dynamic_cast<Executor2*>(obj);
  if (ex2) {
    ex2->extra();
  }
}

int main() {
  // downcasting
  std::unique_ptr<Base> base = std::make_unique<Derived>();

  Derived* derived = dynamic_cast<Derived*>(base.get());
  if (derived) {
    derived->bar();
  }
  // delete derived;  // error double free as derived managed by unique_ptr

  // checking obj types at runtime
  Fake* basePtr = dynamic_cast<Fake*>(base.get());
  if (basePtr) {
    basePtr->zoo();
  }

  std::unique_ptr<Interface> ex1 = std::make_unique<Executor1>();
  std::unique_ptr<Interface> ex2 = std::make_unique<Executor2>();

  execute(ex1.get());
  execute(ex2.get());

  return 0;
}