#include <iostream>

#include "fsmStateController.hpp"

int main(int argc, char const *argv[]) {
  StateController state_conroller;
  state_conroller.handleEvent(Event::activate);
  state_conroller.handleEvent(Event::deactivate);
  return 0;
}