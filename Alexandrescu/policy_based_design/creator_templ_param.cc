#include <cstdlib>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <vector>

template <typename T>
struct NewCreator {
  static T* create() { return new T; }
};

template <typename T>
struct MallocCreator {
  static T* create() {
    void* buf = malloc(sizeof(T));
    if (!buf) {
      return 0;
    }
    return new (buf) T;
  }
};

template <typename T>
class PrototypeCreator {
 private:
  T* prototype_;

 public:
  PrototypeCreator() : prototype_(new T) {}
  PrototypeCreator(T& obj) = delete;
  explicit PrototypeCreator(T* obj) : prototype_(obj) {
    std::cout << "PC ctor obj: " << obj << std::endl;
    static_assert(std::is_pointer<decltype(obj)>::value,
                  "Use only prototype version");
  }
  ~PrototypeCreator() {
    std::cout << "PC: object " << prototype_ << " to been destroyed\n";
    delete prototype_;
  }

  T* create() { return prototype_ ? prototype_->clone() : nullptr; }
  T* get() {
    std::cout << "Called get for: " << prototype_ << std::endl;
    return prototype_;
  }
  void set(T* obj) { prototype_ = obj; }
};

class Widget {
 public:
  Widget() = default;
  Widget(const Widget& other) {
    std::cout << "Widget " << &other << " was copied\n";
  }

  Widget* clone() { return new Widget(*this); }

  friend std::ostream& operator<<(std::ostream& os, const Widget& w) {
    os << "the widget: " << &w;
    return os;
  }
};

class Gadget {
 public:
  Gadget() = default;
  Gadget(const Widget& other) { std::cout << "Gadget was copied\n"; }

  Gadget* clone() { return new Gadget(*this); }
};

// The lib code that uses different Creator policies
template <template <typename> class Creator>
class WidgetManager : public Creator<Widget> {
 private:
  std::vector<Widget*> widgets_;
  // std::vector<Gadget*> gadgets_;
  Creator<Widget> widget_creator_;
  // Creator<Gadget> gadget_creator_;

 public:
  WidgetManager() = default;
  bool add() {
    auto size = widgets_.size();
    widgets_.push_back(widget_creator_.create());
    std::cout << "address of last widget:" << widgets_.back() << std::endl;
    std::cout << "address of it's vector cell:" << &widgets_.back()
              << std::endl;
    return size > widgets_.size();
  }

  void print() {
    for (auto&& item : widgets_) {
      std::cout << item << std::endl;
    }
  }
};

int main(int argc, char const* argv[]) {
  // auto widget = new Widget;
  // PrototypeCreator<Widget> creator(widget);
  // Widget* newWidget = creator.create();
  // if (newWidget) {
  //   std::cout << "Widget created!" << newWidget << " " << widget <<
  //   std::endl;
  // }

  WidgetManager<PrototypeCreator> wm;
  wm.add();
  wm.add();
  wm.add();

  // use only specific methods of PrototypeCreator
  // except common create() for all Creators
  wm.set(new Widget);
  wm.print();

  return 0;
}