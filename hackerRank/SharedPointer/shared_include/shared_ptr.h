#pragma once

template <typename T> class SharedPointer {
public:
  SharedPointer() {
    m_pointer = nullptr;
    m_control = nullptr;
  }

  explicit SharedPointer(T *pointer) {
    try {
      m_control = new ControlBlock(pointer);
      m_pointer = pointer;
    } catch (...) {
      delete pointer;
      throw;
    }
  }
  SharedPointer(const SharedPointer<T> &other)
      : m_pointer(other.m_pointer),
        m_control(ControlBlock::inc(other.m_control)) {}

  SharedPointer(SharedPointer<T> &&other) noexcept
      : m_pointer(other.m_pointer), m_control(other.m_control) {
    other.m_pointer = nullptr;
    other.m_control = nullptr;
  }

  ~SharedPointer() {
    if (m_control) {
      --(*m_control);
      m_pointer = nullptr;
      m_control = nullptr;
    }
  }

  SharedPointer<T> &operator=(const SharedPointer<T> &other) {
    if (this not_eq &other) {
      if (m_control) {
        --(*m_control);
        m_control = other.m_control;
        m_pointer = other.m_pointer;
        if (m_control) {
          ++(*m_control);
        }
      }
    }
    return *this;
  }

  SharedPointer<T> &operator=(SharedPointer<T> &&other) noexcept {
    if (this not_eq &other) {
      if (m_control) {
        --(*m_control);
        m_control = nullptr;
        m_pointer = nullptr;
      }
      this->m_pointer = other.m_pointer;
      this->m_control = other.m_control;
      other.m_pointer = nullptr;
      other.m_control = nullptr;
    }
    return *this;
  }

  bool operator==(const SharedPointer<T> &other) const {
    return m_pointer == other.m_pointer && m_control == other.m_control;
  }

  T &operator*() { return *m_pointer; }

  unsigned int getCount() { return m_control ? m_control->count() : 0; }

  T *get() { return m_pointer; }

  void reset(T *pointer = nullptr) {
    if (m_pointer == pointer) {
      return;
    }

    // old
    T* old_ptr = m_pointer;
    ControlBlock* old_cb = m_control;

    // set new
    m_pointer = pointer;
    m_control = pointer ? new ControlBlock(pointer) : nullptr;
    if (old_cb) {
      (*old_cb)--;
      if (old_cb->count() == 0) {
        delete old_ptr;
        delete old_cb;
      }
    }
  }

private:
  class ControlBlock {
  public:
    ControlBlock() = default;
    explicit ControlBlock(T *m_pointer) : m_pointer(m_pointer) {}
    ~ControlBlock() { delete m_pointer; }

    [[nodiscard]] unsigned int count() const { return m_count; }

    ControlBlock &operator--() {
      if (m_count == 0) {
        return *this;
      }
      --m_count;
      if (m_count == 0) {
        delete this;
      }
      return *this;
    }

    void operator--(int) {
      if (m_count == 0) {
        return;
      }
      --m_count;
      if (m_count == 0) {
        delete this;
      }
    }

    ControlBlock &operator++() {
      ++m_count;
      return *this;
    }

    void operator++(int) { ++m_count; }

    static ControlBlock *inc(ControlBlock *control) {
      if (control) {
        ++(*control);
      }
      return control;
    }

  private:
    T *m_pointer = nullptr;
    unsigned int m_count = 1;
  };
  T *m_pointer = nullptr;
  ControlBlock *m_control = nullptr;
};
