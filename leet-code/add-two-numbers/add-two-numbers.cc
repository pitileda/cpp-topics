#include <cstdint>

struct ListNode {
      int val;
      ListNode *next;
      ListNode() : val(0), next(nullptr) {}
      ListNode(int x) : val(x), next(nullptr) {}
      ListNode(int x, ListNode *next) : val(x), next(next) {}
 };
 
class Solution {
    ListNode* result;
private:
    uint32_t NodeToUint(ListNode* ln){
        uint32_t i = 0;
        uint32_t res = 0;
        ListNode* currentNode = ln;
        while(currentNode->next != nullptr){
            res += currentNode->val;
            currentNode = currentNode->next;
        }
        delete currentNode;
        return res;
    }
    
    ListNode* UintToNode(const uint32_t& i){
        double number = i;
        ListNode* currentNode = result;
        while(number >1 ){
            number = number / 10;
            if(currentNode != nullptr){
                currentNode->val = (int) (number - (uint32_t)(number) )*10;
                currentNode->next = new ListNode();
            }
        }
        if (currentNode) currentNode->next = nullptr;
        return result;
    }
public:
    Solution(): result(){}
    ~Solution(){
        delete result;
    }
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        return UintToNode(NodeToUint(l1) + NodeToUint(l2));
    }
};

int main(int argc, char const *argv[])
{
    ListNode number1_1(2);
    ListNode number1_2(4);
    ListNode number1_3(3);
    number1_1.next = &number1_2;
    number1_2.next = &number1_3;

    ListNode number2_1(5);
    ListNode number2_2(6);
    ListNode number2_3(4);
    number2_1.next = &number2_2;
    number2_2.next = &number2_3;

    Solution s;
    ListNode* result;
    result = s.addTwoNumbers(&number1_1, &number2_1);
    return 0;
}