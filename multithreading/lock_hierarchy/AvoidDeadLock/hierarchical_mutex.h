#include <climits>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
class hierarchical_mutex {
  std::mutex internal_mutex;
  unsigned long hierarchy_value;
  unsigned long prev_hierarchy_value;
  static thread_local unsigned long this_thread_value;

  void check_for_violation() {
    if (this_thread_value <= hierarchy_value) {
      throw std::logic_error(
          "mutex hirarchy violated:" + std::to_string(hierarchy_value) +
          std::string(":") +
          std::to_string(
              std::hash<std::thread::id>{}(std::this_thread::get_id())));
    }
  }

  void update_value() {
    prev_hierarchy_value = this_thread_value;
    this_thread_value = hierarchy_value;
  }

 public:
  explicit hierarchical_mutex(unsigned long value)
      : hierarchy_value(value), prev_hierarchy_value(0) {}

  void lock() {
    check_for_violation();
    internal_mutex.lock();
    update_value();
  }

  void unlock() {
    this_thread_value = prev_hierarchy_value;
    internal_mutex.unlock();
  }

  bool try_lock() {
    check_for_violation();
    if (!internal_mutex.try_lock()) {
      return false;
    }
    update_value();
    return true;
  }
};