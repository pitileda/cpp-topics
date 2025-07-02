#include <gtest/gtest.h>

#include <ostream>

#include "calculator.h"

class CalculatorTest : public testing::Test {
 protected:
  Calculator calc;

  void SetUp() override { calc.reset(); }
  void TearDown() override { std::cout << calc.result() << std::endl; }
};

TEST_F(CalculatorTest, Add) {
  EXPECT_EQ(calc.add(2, 3), 5);
  EXPECT_EQ(calc.result(), 5);
}

TEST_F(CalculatorTest, Subtract) {
  EXPECT_EQ(calc.subtract(2, 3), -1);
  EXPECT_EQ(calc.result(), -1);
}