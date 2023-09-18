#include <gtest/gtest.h>

#include <string>

#include "fifo.h"

namespace fifo::testing {
namespace {
class FifoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    queue_int_.Enqueue(1);
    queue_double_.Enqueue(1.0);
    queue_string_.Enqueue("One");
  }

  Fifo<int> queue_int_;
  Fifo<double> queue_double_;
  Fifo<std::string> queue_string_;
};

TEST_F(FifoTest, Initial) {
  EXPECT_EQ(queue_int_.size(), 1);
  EXPECT_EQ(queue_double_.size(), 1);
  EXPECT_EQ(queue_string_.size(), 1);
}

TEST_F(FifoTest, Add) {
  EXPECT_EQ(queue_int_.size(), 1);
  queue_int_.Enqueue(1);
  EXPECT_EQ(queue_int_.size(), 2);
}

TEST_F(FifoTest, Get) {
  EXPECT_EQ(queue_int_.size(), 1);
  queue_int_.Enqueue(2);
  int* el = queue_int_.Dequeue();
  ASSERT_NE(el, nullptr);
  EXPECT_EQ(*el, 1);
  el = queue_int_.Dequeue();
  ASSERT_NE(el, nullptr);
  EXPECT_EQ(*el, 2);
}

// int main(int argc, char **argv) {
//   ::testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
};  // namespace
}  // namespace fifo::testing
