#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "provider.h"
#include "service.h"

class MockDataProvider : public DataProvider {
 public:
  MOCK_METHOD(int64_t, getData, (int), (override));
};

using ::testing::Exactly;
using ::testing::Return;

TEST(ServiceTest, DoublesDataCorrectly) {
  auto mock = std::make_shared<MockDataProvider>();
  EXPECT_CALL(*mock, getData(42)).Times(Exactly(1)).WillOnce(Return(10));

  Service svc(mock);
  EXPECT_EQ(svc.doubleData(42), 20);
}