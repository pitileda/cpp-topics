#define BOOST_TEST_MODULE unique_ptr_tests

#include "unique_include/unique_ptr.h"
#include <boost/test/unit_test.hpp>
#include <cstdlib>
#include <exception>
#include <new>

struct FailNew {
  FailNew() { failNew = true; }
  ~FailNew() { failNew = false; }
  static bool failNew;
};

bool FailNew::failNew = false;

void *operator new(std::size_t sz) {
  if (!FailNew::failNew) {
    if (sz == 0)
      ++sz; // avoid std::malloc(0) which may return nullptr on success

    if (void *ptr = std::malloc(sz))
      return ptr;
  }
  throw std::bad_alloc{};
}

class TestObject {
public:
  TestObject(bool *b) : destroyed(b) { *destroyed = false; }
  ~TestObject() { *destroyed = true; }
  bool operator==(const TestObject &other) const {
    return destroyed == other.destroyed;
  }
  bool *destroyed;
};

BOOST_AUTO_TEST_SUITE(unique_ptr_tests)

BOOST_AUTO_TEST_CASE(SimpleTest) {
  constexpr int defaultValue = 42;
  auto ptr1 = UniquePointer<int>(new int(defaultValue));
  BOOST_CHECK_EQUAL(*ptr1, defaultValue);
  BOOST_CHECK(ptr1.get() != nullptr);
}

BOOST_AUTO_TEST_CASE(EmptyObject) {
  auto ptr1 = UniquePointer<int>();
  BOOST_CHECK_EQUAL(ptr1.get(), nullptr);
}

BOOST_AUTO_TEST_CASE(MoveConstructor) {
  bool destroyed1 = false;
  UniquePointer<TestObject> ptr1(new TestObject(&destroyed1));
  UniquePointer<TestObject> ptr2 = std::move(ptr1);

  BOOST_CHECK_EQUAL((*ptr2).destroyed, &destroyed1);
  BOOST_CHECK_EQUAL(ptr1.get(), nullptr);
  BOOST_CHECK_EQUAL(destroyed1, false);
}

BOOST_AUTO_TEST_CASE(ReleaseOwnership) {
  bool destroyed = false;
  TestObject *rawPtr = nullptr;
  {
    UniquePointer<TestObject> ptr(new TestObject(&destroyed));
    rawPtr = ptr.release();
    BOOST_CHECK_EQUAL(destroyed, false);
  }
  BOOST_CHECK_EQUAL(destroyed, false);
  delete rawPtr; // Clean up
}

BOOST_AUTO_TEST_CASE(ResetPointer) {
  bool destroyed1 = false;
  bool destroyed2 = false;
  UniquePointer<TestObject> ptr(new TestObject(&destroyed1));
  ptr.reset(new TestObject(&destroyed2));

  BOOST_CHECK_EQUAL(destroyed1, true);
  BOOST_CHECK_EQUAL(destroyed2, false);
  BOOST_CHECK_EQUAL((*ptr).destroyed, &destroyed2);
}

BOOST_AUTO_TEST_CASE(ResetToNull) {
  bool destroyed = false;
  UniquePointer<TestObject> ptr(new TestObject(&destroyed));
  ptr.reset();

  BOOST_CHECK_EQUAL(destroyed, true);
  BOOST_CHECK_EQUAL(ptr.get(), nullptr);
}

BOOST_AUTO_TEST_CASE(MoveAssignment) {
  bool destroyed1 = false;
  bool destroyed2 = false;
  UniquePointer<TestObject> ptr1(new TestObject(&destroyed1));
  UniquePointer<TestObject> ptr2(new TestObject(&destroyed2));
  ptr1 = std::move(ptr2);

  BOOST_CHECK_EQUAL(destroyed1, true);
  BOOST_CHECK_EQUAL(destroyed2, false);
  BOOST_CHECK_EQUAL((*ptr1).destroyed, &destroyed2);
  BOOST_CHECK_EQUAL(ptr2.get(), nullptr);
}

BOOST_AUTO_TEST_CASE(MoveAssignmentWithSelf) {
  bool destroyed1 = false;
  UniquePointer<TestObject> ptr1(new TestObject(&destroyed1));
  ptr1 = std::move(ptr1);
  BOOST_CHECK_EQUAL(destroyed1, false);
  BOOST_CHECK(ptr1.get() != nullptr);
}

BOOST_AUTO_TEST_CASE(ExceptionSafe) {
  bool destroyed = false;
  auto *ptr = new TestObject(&destroyed);
  try {
    auto guard = FailNew();
    auto ptr1 = UniquePointer<TestObject>(ptr);
  } catch (...) {
    BOOST_CHECK_EQUAL(destroyed, true);
  }
  BOOST_CHECK_EQUAL(destroyed, true);
}

BOOST_AUTO_TEST_CASE(SwapTest) {
  bool destroyed1 = false;
  bool destroyed2 = false;
  UniquePointer<TestObject> ptr1(new TestObject(&destroyed1));
  UniquePointer<TestObject> ptr2(new TestObject(&destroyed2));

  ptr1.swap(ptr2);

  BOOST_CHECK_EQUAL((*ptr1).destroyed, &destroyed2);
  BOOST_CHECK_EQUAL((*ptr2).destroyed, &destroyed1);
}

BOOST_AUTO_TEST_SUITE_END()