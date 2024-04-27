#include <iostream>
#include <map>
#include <string>
using namespace std;

struct Node {
  Node *next;
  Node *prev;
  int value;
  int key;
  Node(Node *p, Node *n, int k, int val)
      : prev(p), next(n), key(k), value(val){};
  Node(int k, int val) : prev(NULL), next(NULL), key(k), value(val){};
};

class Cache {

protected:
  map<int, Node *> mp;            // map the key to the node in the linked list
  int cp;                         // capacity
  Node *tail;                     // double linked list tail pointer
  Node *head;                     // double linked list head pointer
  virtual void set(int, int) = 0; // set function
  virtual int get(int) = 0;       // get function
};

class LRUCache : public Cache {
private:
  Node *search(int key) {
    auto it = mp.find(key);
    if (it != mp.end()) {
      return it->second;
    }
    return nullptr;
  }

public:
  LRUCache(int size) {
    cp = size;
    head = nullptr;
    tail = nullptr;
  }

  void set(int k, int v) override {
    if (cp == 0) {
      return;
    }
    // search if key k present in the cache
    Node *found = search(k);
    // if present, make key to be a new head
    if (found) {
      found->value = v;
      if (found->prev) {
        found->prev = found->next;
      }

      if (found->next) {
        found->next->prev = found->prev;
      }

      found->next = head;
      found->prev = nullptr;
    } else { // if no, insert key
      Node *new_head = new Node(nullptr, head, k, v);
      if (head) {
        head->prev = new_head;
      }
      head = new_head;
      if (tail == nullptr) {
        tail = head;
      }
      mp[k] = head;
    }
    // if size > cp remove last el
    if (mp.size() == cp) {
      if (cp == 1) {
        delete head->next;
        head->next = nullptr;
        tail = head;
        mp.erase(k);
        return;
      }
      if (tail) {
        if (tail->prev) {
          tail->prev->next = nullptr;
          Node *new_tail = tail->prev;
          delete tail;
          tail = new_tail;
          mp.erase(k);
        }
      }
    }
  }

  int get(int k) override {
    Node *ret = search(k);
    return ret == nullptr ? -1 : ret->value;
  }
};

int main() {
  int n, capacity, i;
  cin >> n >> capacity;
  LRUCache l(capacity);
  for (i = 0; i < n; i++) {
    string command;
    cin >> command;
    if (command == "get") {
      int key;
      cin >> key;
      cout << l.get(key) << endl;
    } else if (command == "set") {
      int key, value;
      cin >> key >> value;
      l.set(key, value);
    }
  }
  return 0;
}
