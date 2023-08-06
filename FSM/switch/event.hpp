#pragma once
enum class Event { activate, deactivate };

// cout operator for Event
std::ostream &operator<<(std::ostream &os, const Event &event) {
  switch (event) {
    case Event::activate:
      os << "activate";
      break;
    case Event::deactivate:
      os << "deactivate";
      break;
    default:
      break;
  }
  return os;
}