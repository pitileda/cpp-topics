#pragma once
#include "Person.h"
class PersonAddressBuilder;
class PersonJobBuilder;

class PersonBuilderBase {
 protected:
  Person& person;
  explicit PersonBuilderBase(Person& person) : person(person) {}

 public:
  operator Person() const { return person; }

  // facets
  [[nodiscard]] PersonAddressBuilder lives() const;
  [[nodiscard]] PersonJobBuilder works() const;
};

class PersonBuilder : public PersonBuilderBase {
  Person p;

 public:
  PersonBuilder() : PersonBuilderBase(p) {}
};