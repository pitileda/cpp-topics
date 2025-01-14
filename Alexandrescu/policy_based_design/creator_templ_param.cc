#include <cstdlib>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <vector>

// One Policy Creator
// it implements T* create() method
template <typename T>
struct NewCreator {
  static T* create() { return new T; }

  // make destructor protected to disable usage like this
  // typedef WidgetManager<NewCreator> MyWidgetManager;
  // ...
  // MyWidgetManager wm;
  // NewCreator<Widget>* pCreator = &wm; // dubious, but legal
  // delete pCreator; // compiles fine, but has undefined behavior
 protected:
  ~NewCreator() {}
};

// Another One Policy Creator
// it implements T* create() method as well
template <typename T>
struct MallocCreator {
  static T* create() {
    void* buf = malloc(sizeof(T));
    if (!buf) {
      return 0;
    }
    return new (buf) T;
  }

 protected:
  ~MallocCreator() {}
};

// And Even more that 2 Policy creators
// it implements T* create() method
// but also enrich with 2 more methods
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

  T* create() { return prototype_ ? prototype_->clone() : nullptr; }
  T* get() {
    std::cout << "Called get for: " << prototype_ << std::endl;
    return prototype_;
  }
  void set(T* obj) { prototype_ = obj; }

 protected:
  ~PrototypeCreator() {
    std::cout << "PC: object " << prototype_ << " to been destroyed\n";
    delete prototype_;
  }
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

 public:
  WidgetManager() = default;
  ~WidgetManager() {
    for (auto widget : widgets_) {
      delete widget;
    }
  }
  bool add() {
    auto size = widgets_.size();
    widgets_.push_back(this->create());
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
  using Creator<Widget>::get;
  using Creator<Widget>::set;
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

  // The below section is commented after adding
  // protected dtor in base Policy class
  // // use WidgetManager on heap
  // auto pwm = new WidgetManager<PrototypeCreator>();
  // pwm->add();
  // pwm->add();

  // // without protected distructor it is possible to convert
  // // WidgetManager to base class PrototypeCreator<Widget>
  // PrototypeCreator<Widget>* pCreator = pwm;

  // delete pCreator;  // compiles, but has UB even if it works here

  return 0;
}