#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>

#include "provider.h"

class MockDataProvider : public DataProvider {
 public:
  MOCK_METHOD(int64_t, getData, (int), (override));
};