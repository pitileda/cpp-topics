#include <iostream>

enum class State { FULL, LIMITED, BACKGROUND };
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

  State state_;
};

struct Media;

int main() {
  App app;

  app.handle(Event::Activate);
  app.handle(Event::Deactivate);

  Media media;

  media.handle(Event::Activate);
  media.handle(Event::Deactivate);

  return 0;
}