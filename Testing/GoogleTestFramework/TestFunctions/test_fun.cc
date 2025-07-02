#include <gtest/gtest.h>

int add(const int& a, const int& b) { return a + b; }

TEST(PositiveTests, Test01) { EXPECT_EQ(add(12, 14), 26); }

TEST(PositiveTests, Test02) { EXPECT_EQ(add(0, 0), 0); }

TEST(PositiveTests, Test03) { EXPECT_EQ(add(-12, -14), -26); }

TEST(PositiveTests, Test04) { EXPECT_EQ(add(-12, 14), 2); }
