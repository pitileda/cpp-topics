#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <utility>
enum class Event { connect, connected, disconnect, timeout };

inline std::ostream& operator<<(std::ostream& os, const Event& ev) {
  switch (ev) {
    case Event::connect:
      os << "connect" << std::endl;
      break;
    case Event::disconnect:
      os << "diconnect" << std::endl;
      break;
    case Event::connected:
      os << "connected" << std::endl;
      break;
    case Event::timeout:
      os << "timeout" << std::endl;
      break;
  }
  return os;
}

struct State {
  virtual std::unique_ptr<State> on_event(const Event& ev) = 0;
};

struct Idle : public State {
  std::unique_ptr<State> on_event(const Event& ev);
};

struct Connecting : public State {
  std::unique_ptr<State> on_event(const Event& ev);
  uint8_t retries_ = 0;
  uint8_t max_retries_ = 3;
};

struct Connected : public State {
  std::unique_ptr<State> on_event(const Event& ev);
};

std::unique_ptr<State> Idle::on_event(const Event& ev) {
  std::cout << "Idle -> " << ev;
  if (ev == Event::connect) {
    return std::make_unique<Connecting>();
  }
  return nullptr;
}

std::unique_ptr<State> Connecting::on_event(const Event& ev) {
  std::cout << "Connecting -> " << ev;
  switch (ev) {
    case Event::connected:
      return std::make_unique<Connected>();
    case Event::timeout:
      return ++retries_ < max_retries_ ? nullptr : std::make_unique<Idle>();
    case Event::connect:
    case Event::disconnect:
      break;
  }
  return nullptr;
}

std::unique_ptr<State> Connected::on_event(const Event& ev) {
  std::cout << "Connected -> " << ev;
  if (ev == Event::disconnect) {
    return std::make_unique<Idle>();
  }
  return nullptr;
}

struct Bluetooth {
  std::unique_ptr<State> curr_state_ = std::make_unique<Idle>();
  void dispatch(const Event& ev) {
    auto new_state = curr_state_->on_event(ev);
    if (new_state) {
      curr_state_ = std::move(new_state);
    }
  }
  template <typename... Events>
  void send(Events... ev) {
    (dispatch(ev), ...);
  }
};

int main(int argc, char const* argv[]) {
  Bluetooth bl;
  bl.send(Event::connect, Event::timeout, Event::connected, Event::disconnect);
  return 0;
}
