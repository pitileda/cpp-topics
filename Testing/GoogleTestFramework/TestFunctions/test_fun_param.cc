#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include <string>
#include <utility>

template <typename T>
int add(const T& a, const T& b) {
  return a + b;
}

using Input = std::pair<int, int>;
using Value = std::pair<Input, int>;

using InputStr = std::pair<std::string, std::string>;
using ValueStr = std::pair<InputStr, std::string>;

class AdditionTestInt : public ::testing::TestWithParam<Value> {};
class AdditionTestString : public ::testing::TestWithParam<ValueStr> {};

TEST_P(AdditionTestInt, CorrectWithIntegers) {
  auto input = std::get<0>(GetParam());
  int output = std::get<1>(GetParam());

  EXPECT_EQ(add(input.first, input.second), output);
};

// INSTANTIATE_TEST_SUITE_P(
//     AdditionInt, AdditionTestInt,
//     ::testing::Values(std::make_pair(std::make_pair(12, 14), 26),
//                       std::make_pair(std::make_pair(0, 0), 26),
//                       std::make_pair(std::make_pair(100, -200), -100),
//                       std::make_pair(std::make_pair(100000, 900000),
//                       1000000)));

INSTANTIATE_TEST_SUITE_P(AdditionInt, AdditionTestInt,
                         ::testing::Values(Value{{12, 13}, 25},
                                           Value{{12, -13}, -1}));

INSTANTIATE_TEST_SUITE_P(
    AdditionString, AdditionTestString,
    ::testing::Values(ValueStr{InputStr{"aa", "13"}, "aa13"},
                      ValueStr{InputStr{"he", "llo"}, "hello"}));