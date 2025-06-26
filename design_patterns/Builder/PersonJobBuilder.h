#pragma once
#include <string>

#include "PersonBuilder.h"
class PersonJobBuilder : public PersonBuilderBase {
  typedef PersonJobBuilder Self;

 public:
  PersonJobBuilder(Person& person) : PersonBuilderBase(person) {}

  Self& at(std::string company) {
    person.company = company;
    return *this;
  }
  Self& as(std::string position) {
    person.position = position;
    return *this;
  }

  Self& earning(int annual_income) {
    person.annual_income = annual_income;
    return *this;
  }
};