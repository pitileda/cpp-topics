#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>

#include "gmock/gmock.h"
#include "static_foo.h"

class FooClone {
 public:
  virtual int Do(int a) {
    if (a == 0) {
      return 12;
    }
    if (Oops(a) > 10) {
      return 100;
    }

    return 200;
  }

 private:
  virtual int Oops(int b) {
    std::cout << "FooHelper::Oops " << b << std::endl;
    return 34;
  }
};

class MockFooClone : public FooClone {
 public:
  MOCK_METHOD(int, Oops, (int b));
};

TEST(Test, HandleDoTest_Positive) {
  int theA = 0;
  MockFooClone mock;

  // Set expectations for the static methods
  EXPECT_CALL(mock, Oops(theA)).Times(0);

  // Call the static method through the helper class
  mock.Do(theA);
}

TEST(Test, HandleDoTest_Negative) {
  int theA = 1;
  MockFooClone mock;

  // Set expectations for the static methods
  EXPECT_CALL(mock, Oops(theA)).Times(1);

  // Call the static method through the helper class
  mock.Do(theA);
}

TEST(Test, HandleDoTest_OopsReturns60) {
  int theA = 1;
  MockFooClone mock;

  // Set expectations for the static methods
  EXPECT_CALL(mock, Oops(theA)).WillOnce(testing::Return(60));

  // Call the static method through the helper class
  int res = mock.Do(theA);
  EXPECT_EQ(res, 100);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
