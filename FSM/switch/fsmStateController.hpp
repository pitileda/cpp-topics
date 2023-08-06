#include <iostream>

#include "event.hpp"
#include "state.hpp"

// class StateMachine, FSM
class StateController {
 private:
  State state;

  void transition(State newState);

 public:
  StateController(/* args */);
  void handleEvent(Event event);
};

StateController::StateController(/* args */) { state = State::none; }

void StateController::transition(State newState) {
  state = newState;
  std::cout << "State transitioned to " << state << std::endl;
}

void StateController::handleEvent(Event event) {
  switch (event) {
    case Event::activate:
      transition(State::full);
      break;
    case Event::deactivate:
      transition(State::background);
      break;
    default:
      break;
  }
}