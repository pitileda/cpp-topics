#include <cstdint>
#include <string>
struct Foo {
  int x;
  char y;
  std::string z;
};

struct Bar {};

struct Baz : public Bar {
  int x;
};

int* p;
int** pp;
uint64_t ui;
long l;
long long ll;
float f;
double d;

int main(int argc, char const* argv[]) { return sizeof(d); }