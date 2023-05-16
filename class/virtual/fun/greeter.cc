// https://medium.com/@calebleak/fast-virtual-functions-hacking-the-vtable-for-fun-and-profit-25c36409c5e0
#include <iostream>

using namespace std;

class GenericGreeter {
 public:
  virtual void Greet(const char* name) {
    cout << "Hi " << name << "." << endl;
  }

  virtual void Dismiss(const char* name) {
    cout << "Bye " << name << "." << endl;
  }
};

class FriendlyGreeter : public GenericGreeter {
 public:
  virtual void Greet(const char* name) {
    cout << "Hello " << name << "! It's a pleasure to meet you!" << endl;
  }

  virtual void Dismiss(const char* name) {
    cout << "Farewell " << name << "! Until later!" << endl;
  }
};

typedef void (GreetFn)(void*, const char* name);
struct GenericGreeter_VTable {
  GreetFn* greet;
  GreetFn* dismiss;
};

// function to find a VTable pointer
GenericGreeter_VTable* GetVTable(GenericGreeter* obj) {
  GenericGreeter_VTable** vtable_ptr = (GenericGreeter_VTable**)obj;
  return *(vtable_ptr);
}

int main() {
  FriendlyGreeter* friendlyGreeter = new FriendlyGreeter;
  GenericGreeter* genericGreeter = (GenericGreeter*)friendlyGreeter;

  friendlyGreeter->Greet("Bob");
  genericGreeter->Greet("Alice");

  GetVTable(friendlyGreeter)->greet(friendlyGreeter, "Bob");
  GetVTable(friendlyGreeter)->dismiss(friendlyGreeter, "Bob");

  GetVTable(genericGreeter)->greet(friendlyGreeter, "Alice");
  GetVTable(genericGreeter)->dismiss(friendlyGreeter, "Alice");

  delete friendlyGreeter;

  return 0;
}