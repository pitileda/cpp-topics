#include <concepts>
#include <iostream>
#include <type_traits>

// C++17
template <typename T, typename = typename std::enable_if_t<
                          std::is_arithmetic_v<T> || std::is_integral_v<T>>>
struct Node {
  T data;
  Node* next;
  Node* prev;
  Node(T val) : data(val), next(nullptr), prev(nullptr) {}
};

// C++20
template <typename T>
concept arithmetic = std::integral<T> || std::floating_point<T>;

template <arithmetic T>
class List {
 private:
  Node<T>* head;

 public:
  List() : head(nullptr) {}

  ~List() {
    Node<T>* curr = head;
    while (curr != nullptr) {
      Node<T>* temp = curr;
      curr = curr->next;
      delete temp;
    }
  }

  bool insert(int pos, const int& value) {
    if (pos < 0) return false;
    Node<T>* after = head;
    Node<T>* before = nullptr;
    Node<T>* newNode = new Node<T>(value);
    int counter{0};
    while (after != nullptr) {
      if (counter == pos) {
        break;
      }
      before = after;
      after = after->next;
      counter++;
    }
    newNode->prev = before;
    newNode->next = after;
    if (before) {
      before->next = newNode;
    }
    if (after) {
      after->prev = newNode;
    }
    if (pos == 0) {
      head = newNode;
    }
    return true;
  }

  void erase(Node<T>* el) {
    if (!el) return;
    el->prev->next = el->next;
    el->next->prev = el->prev;
    delete el;
  }

  void erase(int pos) {
    Node<T>* curr = head;
    int counter{0};
    while (curr != nullptr) {
      if (counter == pos) {
        break;
      }
      counter++;
      curr = curr->next;
    }
    if (curr) {
      if (curr->prev) {
        curr->prev->next = curr->next;
      }
      if (curr->next) {
        curr->next->prev = curr->prev;
      }
      if (pos == 0) {
        if (curr->next) {
          head = curr->next;
        } else {
          head = nullptr;
        }
      }
      delete curr;
    }
  }

  void print() {
    Node<T>* curr = head;
    while (curr != nullptr) {
      std::cout << curr->data << "|" << std::endl;
      curr = curr->next;
    }
  }
};

int main() {
  List<int> lst;
  lst.insert(0, 5);
  lst.print();
  std::cout << "Wow\n";
  lst.insert(0, 10);
  lst.print();
  std::cout << "Hello\n";
  lst.insert(1, 15);
  lst.print();
  std::cout << "Last\n";
  lst.insert(3, 45);
  lst.print();

  std::cout << "erase 3\n";
  lst.erase(3);
  lst.print();

  std::cout << "erase 0\n";
  lst.erase(0);
  lst.print();

  std::cout << "erase 0\n";
  lst.erase(0);
  lst.print();

  std::cout << "erase 0\n";
  lst.erase(0);
  lst.print();

  std::cout << "erase 0\n";
  lst.erase(0);
  lst.print();

  // List<std::string> lst_str; // compile time error

  return 0;
}