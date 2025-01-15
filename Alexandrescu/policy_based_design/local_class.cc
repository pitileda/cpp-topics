// Local class to make adapter
// that wrap object to another one
// that is compatible with Interface

class Interface {
 public:
  virtual bool fun() = 0;
};

class My {
 public:
  int make_happy(const int& points) { return points; }
};

template <typename ObjectType, typename ArgType>
Interface* ToInterface(const ObjectType& obj, const ArgType& arg) {
  class Local : public Interface {
   public:
    Local(const ObjectType& obj, const ArgType& arg) : obj_(obj), arg_(arg) {}
    virtual bool fun() { return static_cast<bool>(obj_.make_happy(arg_)); }

   private:
    ObjectType obj_;
    ArgType arg_;
  };
  return new Local(obj, arg);
}

int main() {
  My my_object;
  Interface* interface_my_object = ToInterface(my_object, 1);
  return interface_my_object->fun();
}