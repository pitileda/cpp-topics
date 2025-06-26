#pragma once
#include <string>
#include <utility>

#include "PersonBuilder.h"

class PersonAddressBuilder : public PersonBuilderBase {
  typedef PersonAddressBuilder Self;

 public:
  explicit PersonAddressBuilder(Person& person) : PersonBuilderBase(person) {}

  Self& has_name(std::string name) {
    person.name = std::move(name);
    return *this;
  }

  Self& at(std::string address) {
    person.address = std::move(address);
    return *this;
  }

  Self& in(std::string city) {
    person.city = std::move(city);
    return *this;
  }

  Self& has(std::string name) {
    person.name = std::move(name);
    return *this;
  }
};