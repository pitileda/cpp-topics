#include <gtest/gtest.h>

#include <string>

#include "parser.h"

using namespace std;

TEST(Positive, SplitMethodPositive) {
  string s{"S;C;LargeSoftwareBook"};
  s = parseInputLine('S', 'C', s);
  EXPECT_EQ("large software book", s);
}

TEST(Positive, SplitMethodPositiveWithBrackets) {
  string s{"S;M;plasticCup()"};
  s = parseInputLine('S', 'C', s);
  EXPECT_EQ("plastic cup", s);
}

TEST(Negative, SplitMethodNegative) {
  string s{"S;C;"};
  s = parseInputLine('S', 'C', s);
  EXPECT_EQ("", s);
}
