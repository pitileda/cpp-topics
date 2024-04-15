#pragma once

#include <cstddef>
#include <utility>

enum class Code { ErrorSize, ErrorIndex, Success };

namespace sc {

namespace {

template <typename T, size_t N> class StaticArray {
public:
  StaticArray(const std::initializer_list<T> &list) {
    size_t i = 0;
    for (auto &&e : list) {
      data_[i++] = e;
    }
  }
  StaticArray() = default;
  size_t size() const { return N; }
  T &operator[](const size_t i) { return data_[i]; }

private:
  T data_[N];
};

} // namespace

template <typename T, size_t N> class Queue {
public:
  Queue(const std::initializer_list<T> &list) : data_(list) {}
  Queue() = default;
  Code push(const T &el) {
    if (capacity_ == 0) {
      data_[tail_] = el;
      capacity_++;
      return Code::Success;
    }
    if (capacity_ < data_.size()) {
      ++tail_;
      if (tail_ < data_.size()) {
        data_[tail_] = el;
        capacity_++;
        return Code::Success;
      }
      tail_ = 0;
      data_[tail_] = el;
      capacity_++;
    }
    return Code::ErrorSize;
  }
  void pop() {
    if (capacity_ == 0) {
      return;
    }
    ++head_;
    if (head_ >= data_.size()) {
      head_ = 0;
    }
    --capacity_;
  }

  std::pair<T &, Code> get(const size_t index) {
    if ((head_ < tail_ && index >= head_ && index <= tail_) ||
        (head_ > tail_ && !(index > tail_ && index < head_))) {
      return {data_[head_ + index], Code::Success};
    }
    return {data_[0], Code::ErrorIndex};
  }

  size_t size() const { return capacity_; }
  bool empty() const { return capacity_ == 0 ? true : false; }

private:
  StaticArray<T, N> data_;
  size_t head_ = 0;
  size_t tail_ = 0;
  size_t capacity_ = 0;
};

template <typename T, size_t N> class Stack {
public:
  Stack(const std::initializer_list<T> &list) : data_(list) {}
  Stack() = default;
  void push(const T &el) {
    if (size_ < N) {
      data_[top_] = el;
      top_++;
      size_++;
    }
  }
  void pop() {
    if (size_ != 0) {
      --top_;
      --size_;
    }
  }
  bool empty() { return size_ == 0 ? true : false; }
  size_t size() { return size_; }

  std::pair<T &, Code> get(const size_t index) {
    if (index < top_) {
      return {data_[index], Code::Success};
    }
    return {data_[0], Code::ErrorIndex};
  }

private:
  StaticArray<T, N> data_;
  size_t size_ = 0;
  size_t top_ = 0;
};
} // namespace sc
