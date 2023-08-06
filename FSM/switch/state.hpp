#pragma once
enum class State { none, background, limited, full };

// cout operator for State
std::ostream &operator<<(std::ostream &os, const State &state) {
  switch (state) {
    case State::none:
      os << "none";
      break;
    case State::background:
      os << "background";
      break;
    case State::limited:
      os << "limited";
      break;
    case State::full:
      os << "full";
      break;
    default:
      break;
  }
  return os;
}