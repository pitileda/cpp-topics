#include "gtest/gtest.h"

TEST(HelloTestSuit, HelloTest) {
  EXPECT_STRNE("Hello", "World");
  EXPECT_EQ(7 * 8, 56);
}