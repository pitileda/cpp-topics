#include <algorithm>

class Node {
 public:
  int val;
  Node* next;

  Node(int initialVal) {
    val = initialVal;
    next = nullptr;
  }
};

int longestStreak(Node* head) {
  int max_length = 0;
  Node* current = head;
  Node* previous = nullptr;

  int counter = 0;
  while (current != nullptr) {
    if (previous == nullptr || current->val == previous->val) {
      counter++;
    } else {
      counter = 1;
    }
    max_length = std::max(max_length, counter);
    previous = current;
    current = current->next;
  }
  return max_length;
}

int main() {
  Node a(5);
  Node b(5);
  Node c(7);
  Node d(7);
  Node e(7);
  Node f(6);

  a.next = &b;
  b.next = &c;
  c.next = &d;
  d.next = &e;
  e.next = &f;
  return longestStreak(&a);
}