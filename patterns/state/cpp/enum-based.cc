#include <iostream>

enum class State { FULL, LIMITED, BACKGROUND, NONE };
enum class Event { Activate, Deactivate };

struct App {
  void handle(const Event& e) {
    switch (e) {
      case Event::Activate: {
        // do act
        state_ = State::FULL;
        break;
      }
      case Event::Deactivate: {
        // do deact
        state_ = State::BACKGROUND;
        break;
      }
      default:
        std::cout << "Ignoring...\n";
    }
  }

  State getLevel() { return state_; }

  State state_ = State::NONE;
};

std::ostream& operator<<(std::ostream& os, const State& s) {
  if (s == State::NONE) {
    os << "NONE";
  }
  if (s == State::BACKGROUND) {
    os << "BACKGROUND";
  }
  if (s == State::LIMITED) {
    os << "LIMITED";
  }
  if (s == State::FULL) {
    os << "FULL";
  }
  return os;
}

struct Media;

int main() {
  App app;

  std::cout << app.getLevel() << std::endl;
  app.handle(Event::Activate);
  std::cout << app.getLevel() << std::endl;
  app.handle(Event::Deactivate);
  std::cout << app.getLevel() << std::endl;
  app.handle(Event::Activate);
  std::cout << app.getLevel() << std::endl;

  // Media media;

  // media.handle(Event::Activate);
  // media.handle(Event::Deactivate);

  return 0;
}