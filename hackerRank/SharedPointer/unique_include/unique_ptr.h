#pragma once

template <typename T> class UniquePointer {
public:
  UniquePointer() { m_pointer = nullptr; }
  explicit UniquePointer(T *pointer) : m_pointer(pointer) {}
  UniquePointer(const UniquePointer<T> &other) = delete;
  UniquePointer(UniquePointer<T> &&other) noexcept
      : m_pointer(other.release()) {}
  ~UniquePointer() { delete m_pointer; }

  UniquePointer<T> &operator=(const UniquePointer<T> &other) = delete;
  UniquePointer<T> &operator=(UniquePointer<T> &&other) noexcept {
    if (this not_eq &other) {
      reset(other.release());
    }
    return *this;
  }
  bool operator==(const UniquePointer &other) const = delete;

  T &operator*() const { return *m_pointer; }
  T *operator->() const { return m_pointer; }
  T *get() const noexcept { return m_pointer; }
  explicit operator bool() const noexcept { return m_pointer != nullptr; }
  T *release() noexcept {
    T *temp = m_pointer;
    m_pointer = nullptr;
    return temp;
  }

  void reset(T *pointer = nullptr) noexcept {
    const T *old = m_pointer;
    m_pointer = pointer;
    delete old;
  }

  void swap(UniquePointer &other) noexcept {
    T *temp = m_pointer;
    m_pointer = other.m_pointer;
    other.m_pointer = temp;
  }

private:
  T *m_pointer = nullptr;
};