#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

class HTMLBuilder;
class HTMLElement {
  friend class HTMLBuilder;
  std::string name;
  std::string text;
  std::vector<HTMLElement> children;
  int padding = 2;
  HTMLElement(std::string name, std::string text)
      : name(std::move(name)), text(std::move(text)) {}

 public:
  [[nodiscard]] std::string str(int start_padding) const {
    std::ostringstream oss;
    std::string _(padding * start_padding, ' ');
    oss << _ << "<" << name << ">" << std::endl;
    if (!text.empty()) {
      oss << std::string(padding * (start_padding + 1), ' ') << text
          << std::endl;
    }
    for (const auto& element : children) {
      oss << element.str(start_padding + 1);
    }
    oss << _ << "</" << name << ">" << std::endl;
    return oss.str();
  }
  static HTMLBuilder create(std::string root_name);
};

class HTMLBuilder {
  HTMLElement root;

 public:
  explicit HTMLBuilder(std::string name) : root("", "") {
    root.name = std::move(name);
  }
  HTMLBuilder& add_child(std::string child_name, std::string text) {
    HTMLElement e(std::move(child_name), std::move(text));
    root.children.emplace_back(std::move(e));
    return *this;
  }

  [[nodiscard]] std::string str(int start_padding = 0) const {
    return root.str(start_padding);
  }

  HTMLElement build() { return root; }
  explicit operator HTMLElement() const { return root; }
};

inline HTMLBuilder HTMLElement::create(std::string root_name) {
  return HTMLBuilder(std::move(root_name));
}

int main() {
  HTMLBuilder builder("ul");
  builder.add_child("li", "Ihor").add_child("li", "Olena");
  std::cout << builder.str();
  HTMLElement root = static_cast<HTMLElement>(HTMLElement::create("ul")
                                                  .add_child("li", "Ihor")
                                                  .add_child("li", "Polina"));
  auto Olena = static_cast<HTMLElement>(HTMLElement::create("p").build());
  return 0;
}