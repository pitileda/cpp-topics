#include <iostream>
#include <memory>
#include <tuple>
#include <utility>
#include <variant>

template <typename... States>
class StateMachine {
 public:
  template <typename State>
  void transitionTo() {
    currentState = std::move(std::get<std::unique_ptr<State>>(states));
  }

  template <typename Event>
  void handle(const Event& event) {
    auto passEventToState = [this, &event](auto statePtr) {
      statePtr->handle(event).execute(*this);
    };
    std::visit(passEventToState, std::move(currentState));
  }

 private:
  std::tuple<std::unique_ptr<States>...> states;
  std::variant<std::unique_ptr<States>...> currentState{
      std::move(std::get<0>(states))};
};

template <typename State>
struct TransitionTo {
  template <typename Machine>
  void execute(Machine& machine) {
    machine.template transitionTo<State>();
  }
};

struct Nothing {
  template <typename Machine>
  void execute(Machine&) {}
};

struct OpenEvent {};

struct CloseEvent {};

struct ClosedState;
struct OpenState;

struct ClosedState {
  TransitionTo<OpenState> handle(const OpenEvent&) const {
    std::cout << "Opening the door..." << std::endl;
    return {};
  }

  Nothing handle(const CloseEvent&) const {
    std::cout << "Cannot close. The door is already closed!" << std::endl;
    return {};
  }
};

struct OpenState {
  Nothing handle(const OpenEvent&) const {
    std::cout << "Cannot open. The door is already open!" << std::endl;
    return {};
  }

  TransitionTo<ClosedState> handle(const CloseEvent&) const {
    std::cout << "Closing the door..." << std::endl;
    return {};
  }
};

using Door = StateMachine<ClosedState, OpenState>;

int main() {
  Door door;

  door.handle(OpenEvent{});
  door.handle(CloseEvent{});

  door.handle(CloseEvent{});
  door.handle(OpenEvent{});

  return 0;
}