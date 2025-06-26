// compile with clang-18, built from source
// g++ is not ready for now
import std;

// std::generator is not yet ready in clang
// std::generator<int> generate_numbers(int start, int end) {
//     for (int i = start; i <= end; ++i) {
//         co_yield i;
//     }
// }

// below is c++20 vesrion

int main() {
  std::println("Hello, world");
  return 0;
}