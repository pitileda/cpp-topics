#include <iostream>

#include "Person.h"
#include "PersonAddressBuilder.h"
#include "PersonJobBuilder.h"
int main(int argc, char *argv[]) {
  Person p = Person::create()
                 .lives()
                 .has_name("Ihor")
                 .at("Plebiscytowa")
                 .in("Opole")
                 .works()
                 .at("Lux")
                 .earning(100000)
                 .as("develoeper");
  std::cout << p;
}
