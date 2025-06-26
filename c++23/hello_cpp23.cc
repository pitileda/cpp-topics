// compile with clang-18, built from source
// g++ is not ready for now
// https://0xstubs.org/using-the-c23-std-module-with-clang-18/
// https://discourse.llvm.org/t/llvm-discussion-forums-libc-c-23-module-installation-support/77087/16
import std;

int main() {
  std::println("Hello, world");
  std::vector<int> res{0, 1};
  return res[1];
}