#pragma once
#include <functional>
#include <map>

enum event_t {
  EV_NOTHING = 0,
  EV_START_SENSORS_CMD,
  EV_STOP_SENSORS_CMD,
  /* etc. */
};

struct state_machine {
  using event_handler_t =
      std::function<bool(event_t)>;  // return true if handled
  void register_handler(event_t event, event_handler_t handler) {
    handler_map[event] = handler;
  }

  bool handle_event(event_t event) {
    auto iter = handler_map.find(event);
    if (iter == handler_map.end())
      return false;
    else
      return iter->second(event);  // call handler, return its result
  }

  std::map<event_t, event_handler_t> handler_map;
};