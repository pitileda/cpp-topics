// compile with clang-18, built from source
// g++ is not ready for now
import std;

int main() {
  std::println("Hello, world");
  constexpr int x = 42;
  std::println("int x: {}", x);
  std::println("int x as bool: {:b}", x);
  std::println("int x as hex: {:#X}", x);
  std::println("int x as octal: {:o}", x);
  std::println("Right aligned: {:>5}", x);
  std::println("Left aligned: {:<5}", x);
  std::println("Center with dots: {:^7}", x);
  std::println("Positive: {:+}", x);
  std::println("Negative: {:-}", x * (-1));
  // constexpr int y = x * 10000000;
  // std::println("Thousands separator: {:,}", y);

  std::string text = "Hello, World!";
  std::println("Truncated: {:.5}", text);
  std::println("Right aligned: {:>20}", text);
  return 0;
}