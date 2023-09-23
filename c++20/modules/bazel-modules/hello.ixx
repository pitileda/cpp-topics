export module hello;

import<iostream>;
import<string_view>;

export inline void say_hello(std::string_view const& name) {
  std::cout << "Hello " << name << std::endl;
}