#include "bar.h"

#include <gtest/gtest.h>

#include "gmock/gmock.h"

#ifdef UNIT_TESTS
class MockBar : public Bar {
 public:
  MOCK_METHOD(int, Oops, (int b));
};
#endif

class BarTest : public ::testing::Test {
 protected:
  MockBar bar_;
};

TEST_F(BarTest, HandleDoTest_Positive) {
  int theA = 0;

  // Set expectations for the static methods
  EXPECT_CALL(bar_, Oops(theA)).Times(0);
  bar_.Do(theA);
}

TEST_F(BarTest, HandleDoTest_Positive2) {
  int theA = 1;

  // Set expectations for the static methods
  EXPECT_CALL(bar_, Oops(theA)).Times(1);
  bar_.Do(theA);
}

TEST_F(BarTest, HandleDoTest_Positive3) {
  int theA = 1;

  // Set expectations for the static methods
  EXPECT_CALL(bar_, Oops(theA)).WillOnce(testing::Return(6));
  auto res = bar_.Do(theA);
  EXPECT_EQ(res, 200);
}

TEST_F(BarTest, HandleDoTest_Positive4) {
  int theA = 1;

  // Set expectations for the static methods
  EXPECT_CALL(bar_, Oops(theA)).WillOnce(testing::Return(11));
  auto res = bar_.Do(theA);
  EXPECT_EQ(res, 100);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
