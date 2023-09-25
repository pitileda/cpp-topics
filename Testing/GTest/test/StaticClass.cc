#include <cstdint>

#include "gmock/gmock.h"

class Foo {
 public:
  static void Do(int a) {
    if (a > 0) {
      return;
    }
    Oops(a);
  }

 private:
  static void Oops(int b) { std::cout << "Foo::Oops " << b << std::endl; }
};

class FooClone {
 public:
  virtual void Do(int a) {
    if (a > 0) {
      return;
    }
    Oops(a);
  }
  virtual void Oops(int b) {
    std::cout << "FooHelper::Oops " << b << std::endl;
  }

 private:
  static bool privateOopsCalled;
};

class MockFooClone : public FooClone {
 public:
  MOCK_METHOD(void, Oops, (int b));
};

TEST(Test, HandleDoTest_Positive) {
  int theA = 0;
  MockFooClone mock;

  // Set expectations for the static methods
  EXPECT_CALL(mock, Oops(theA)).Times(1);

  // Call the static method through the helper class
  mock.Do(theA);
}

TEST(Test, HandleDoTest_Negative) {
  int theA = 1;
  MockFooClone mock;

  // Set expectations for the static methods
  EXPECT_CALL(mock, Oops(theA)).Times(0);

  // Call the static method through the helper class
  mock.Do(theA);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}
