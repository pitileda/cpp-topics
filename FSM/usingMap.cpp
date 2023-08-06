#include <algorithm>
#include <iostream>

#include "map.hpp"

int main() {
  struct X {
    bool nothing_handler([[maybe_unused]] event_t nothing) const {
      std::cout << "X::nothing_handler\n";
      return true;
    }
  };

  X x;

  state_machine sm;
  // register two event handlers
  sm.register_handler(
      EV_NOTHING,
      std::bind(&X::nothing_handler, std::addressof(x), std::placeholders::_1));
  sm.register_handler(EV_START_SENSORS_CMD, []([[maybe_unused]] event_t start) {
    std::cout << "starting sensors now\n";
    return true;
  });

  // handle some events
  sm.handle_event(EV_START_SENSORS_CMD);
  sm.handle_event(EV_NOTHING);
  std::cout << "handled? " << std::boolalpha
            << sm.handle_event(EV_STOP_SENSORS_CMD)
            << '\n';  // false (unhandled)
}