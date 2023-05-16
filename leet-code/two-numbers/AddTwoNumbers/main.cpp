#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <forward_list>

struct ListNode {
  int val;
  ListNode* next;
  ListNode() : val(0), next(nullptr) {}
  ListNode(int x) : val(x), next(nullptr) {}
  ListNode(int x, ListNode* next) : val(x), next(next) {}
};

class Solution {
 public:
  ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    ListNode* dummyHead = new ListNode(0);
    ListNode* curr = dummyHead;
    int carry = 0;
    while (l1 != NULL || l2 != NULL || carry != 0) {
      int x = l1 ? l1->val : 0;
      int y = l2 ? l2->val : 0;
      int sum = carry + x + y;
      carry = sum / 10;
      curr->next = new ListNode(sum % 10);
      curr = curr->next;
      l1 = l1 ? l1->next : nullptr;
      l2 = l2 ? l2->next : nullptr;
    }
    return dummyHead->next;
  }
};

std::string str(ListNode* ln) {
  std::string res;
  if (ln == nullptr) {
    return res;
  }
  while (ln->next != nullptr) {
    res += std::to_string(ln->val);
    res += ' ';
    ln++;
  }
  res += std::to_string(ln->val);
  return res;
}

int main(int argc, char const* argv[]) {
  std::vector<ListNode> number1;
  number1.push_back(2);
  number1.push_back(4);
  number1.push_back(3);
  number1[0].next = &number1[1];
  number1[1].next = &number1[2];

  std::vector<ListNode> number2;
  number2.push_back(5);
  number2.push_back(6);
  number2.push_back(4);
  number2[0].next = &number2[1];
  number2[1].next = &number2[2];

  Solution s;
  ListNode* result;
  result = s.addTwoNumbers(&number1[0], &number2[0]);
  std::cout << str(result);
  return result->val;
}
