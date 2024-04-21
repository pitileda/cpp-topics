// https://www.vishalchovatiya.com/state-design-pattern-in-modern-cpp/
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <variant>

// Events
struct EventConnect {
  std::string address_;
};

struct EventConnected {};
struct EventDisconnect {};
struct EventTimeout {};

using Event =
    std::variant<EventConnect, EventConnected, EventDisconnect, EventTimeout>;

// States

struct Idle {};
struct Connecting {
  std::string address_;
  uint8_t retries_ = 0;
  static const uint8_t max_retries_ = 3;
};
struct Connected {};

using State = std::variant<Idle, Connected, Connecting>;

// https://stackoverflow.com/questions/62355613/stdvariant-cout-in-c
struct StateCouter {
  std::string operator()(const Idle& s) const { return std::string("Idle"); }
  std::string operator()(const Connecting& s) const {
    return std::string("Connecting");
  }
  std::string operator()(const Connected& s) const {
    return std::string("Connected");
  }
};

// Rules
struct Rules {
  std::optional<State> operator()(Idle& s, const EventConnect& e) {
    std::cout << "Idle -> Connect, " << typeid(s).name() << std::endl;
    return Connecting{e.address_};
  }

  std::optional<State> operator()(Connecting& s, const EventConnected& e) {
    std::cout << "Connecting -> Connected, " << typeid(s).name() << std::endl;
    return Connected{};
  }

  std::optional<State> operator()(Connecting& s, const EventTimeout& e) {
    std::cout << "Connecting -> Timeout, " << typeid(s).name() << std::endl;
    return ++s.retries_ < Connecting::max_retries_
               ? std::nullopt
               : std::optional<State>(Idle{});
  }

  std::optional<State> operator()(Connected& s, const EventDisconnect& e) {
    std::cout << "Connected -> Disconnect, " << typeid(s).name() << std::endl;
    return Idle{};
  }

  template <typename State_t, typename Event_t>
  std::optional<State> operator()(State_t& s, const Event_t& e) {
    std::cout << "Unknown" << typeid(s).name() << std::endl;
    return std::nullopt;
  }
};

template <typename StateVariant, typename EventVariant, typename Rules>
struct Bluetooth {
  StateVariant curr_state_;

  void dispatch(const EventVariant& ev) {
    std::optional<StateVariant> new_state =
        std::visit(Rules{}, curr_state_, ev);
    if (new_state) {
      curr_state_ = *std::move(new_state);
    }
    std::cout << "BT state is: " << std::visit(StateCouter(), curr_state_)
              << std::endl;
  }

  template <typename... Events>
  void send(const Events&... ev) {
    (dispatch(ev), ...);
  }
};

int main(int argc, char const* argv[]) {
  Bluetooth<State, Event, Rules> bl;
  bl.send(EventTimeout(), EventConnect{"AA:BB:CC:DD:FF"}, EventTimeout(),
          EventConnected(), EventTimeout(), EventDisconnect());
  return EXIT_SUCCESS;
}
