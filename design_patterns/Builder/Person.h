#pragma once
#include <ostream>
#include <string>

class PersonBuilder;
class PersonAddressBuilder;
class PersonJobBuilder;
class Person {
  friend std::ostream& operator<<(std::ostream& os, const Person& obj) {
    return os << "name: " << obj.name << " address: " << obj.address
              << " city: " << obj.city << " company: " << obj.company
              << " position: " << obj.position
              << " annual_income: " << obj.annual_income;
  }
  std::string name, address, city;
  std::string company, position;
  int annual_income;

  Person() {}

 public:
  ~Person() = default;
  static PersonBuilder create();
  friend class PersonBuilder;
  friend class PersonAddressBuilder;
  friend class PersonJobBuilder;
};