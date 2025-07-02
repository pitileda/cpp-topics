#pragma once

#include <cstdint>
class Calculator {
 public:
  void reset() { result_ = 0; }

  int64_t add(const int64_t& left, const std::int64_t& right) {
    return result_ = left + right;
  }

  int64_t subtract(const int64_t& left, const int64_t& right) {
    return result_ = left - right;
  }

  int64_t result() const { return result_; }

 private:
  int64_t result_;
};