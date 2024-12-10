// https://www.vishalchovatiya.com/state-design-pattern-in-modern-cpp/
#include <cstdint>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
/*------------------------------- Events ------------------------------------*/
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

/*------------------------------- States ------------------------------------*/
struct State {
  virtual std::unique_ptr<State> on_event(const Event& ev) = 0;
  virtual std::string get_name() const { return std::string("DefaultState"); }
};

struct Idle : public State {
  std::unique_ptr<State> on_event(const Event& ev);
  virtual std::string get_name() const { return std::string("Idle"); }
};

struct Connecting : public State {
  std::unique_ptr<State> on_event(const Event& ev);
  uint8_t retries_ = 0;
  uint8_t max_retries_ = 3;
  virtual std::string get_name() const { return std::string("Connecting"); }
};

struct Connected : public State {
  std::unique_ptr<State> on_event(const Event& ev);
  virtual std::string get_name() const { return std::string("Connected"); }
};

inline std::ostream& operator<<(std::ostream& os, const State& state) {
  os << state.get_name();
  return os;
}

/*------------------------------- Transitions -------------------------------*/
std::unique_ptr<State> Idle::on_event(const Event& ev) {
  std::cout << "In \"Idle\" state received event: " << ev;
  if (ev == Event::connect) {
    return std::make_unique<Connecting>();
  }
  return nullptr;
}

std::unique_ptr<State> Connecting::on_event(const Event& ev) {
  std::cout << "In \"Connecting\" state received event: " << ev;
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
  std::cout << "In \"Connected\" state received event: " << ev;
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
    std::cout << "BT state is: " << *curr_state_ << std::endl;
  }
  template <typename... Events>
  void process(Events... ev) {
    (dispatch(ev), ...);
  }
};

int main(int argc, char const* argv[]) {
  Bluetooth bl;
  bl.process(Event::timeout, Event::connect, Event::timeout, Event::connected,
             Event::disconnect);
  return 0;
}
