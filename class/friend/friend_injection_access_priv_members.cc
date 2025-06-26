class foo {
 public:
  int get() const { return data; }

 private:
  int data;
};

template <int foo::* Ptr>
int& get_data(foo& f) {
  return f.*Ptr;
}

template <int foo::* Ptr>
struct foo_access {
  friend int& get_data(foo& f) { return f.*Ptr; }
};

template struct foo_access<&foo::data>;
int& get_data(foo&);

int main() {
  foo f{};
  get_data(f) = 42;  // access private data member
  return f.get();
}