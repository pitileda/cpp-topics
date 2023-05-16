#include <iostream>

using namespace std;

// Our Greet function type.
typedef void (GreetFn)(void*, const char* name);

struct GenericGreeter_VTable {
  GreetFn* greet;
};

// Forward declare VTable instances.
extern GenericGreeter_VTable generic_vtable;
extern GenericGreeter_VTable friendly_vtable;

class GenericGreeter {
 public:
  GenericGreeter_VTable* vtable;

  GenericGreeter() {
    vtable = &generic_vtable;
  }

  static void GreetGeneric(void* _this, const char* name) {
    cout << "Hi " << name << "." << endl;
  }
};

class FriendlyGreeter : public GenericGreeter {
 public:
  FriendlyGreeter() {
    vtable = &friendly_vtable;
  }

  static void GreetFriendly(void* _this, const char* name) {
    cout << "Hello " << name << "! It's a pleasure to meet you!" << endl;
  }
};

// Create the static VTable instances
GenericGreeter_VTable generic_vtable = {
  (GreetFn*)&GenericGreeter::GreetGeneric
};

GenericGreeter_VTable friendly_vtable = {
  (GreetFn*)&FriendlyGreeter::GreetFriendly
};

int main() {
  FriendlyGreeter* friendlyGreeter = new FriendlyGreeter;
  GenericGreeter* genericGreeter = (GenericGreeter*)friendlyGreeter;

  friendlyGreeter->vtable->greet(friendlyGreeter, "Bob");
  genericGreeter->vtable->greet(genericGreeter, "Alice");

  delete friendlyGreeter;

  return 0;
}
