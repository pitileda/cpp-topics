#pragma once

#include <cstdint>

class DataProvider {
 public:
  virtual ~DataProvider() = default;
  virtual auto getData(int id) -> int64_t = 0;
};