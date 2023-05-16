#include <iomanip>
#include <iostream>

#include "trie.h"

int main(int, char**) {
  ikov::Trie t;

  t.insert("Hello");
  std::cout << "trie is: [Hello]\n";
  std::cout << "trie.search(Hello): " << std::boolalpha << t.search("Hello")
            << std::endl;
  std::cout << "trie.search(Hell): " << std::boolalpha << t.search("Hell")
            << std::endl;
  std::cout << "trie.startWith(He): " << std::boolalpha << t.startWith("He")
            << std::endl;
  return 0;
}
