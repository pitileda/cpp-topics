#pragma once

#include <iostream>

class App {
 public:
  App() = default;
  ~App() = default;

  void onActivate() { std::cout << "App::onActivate()" << std::endl; }
  void onDeactivate() { std::cout << "App::onDeactivate()" << std::endl; }
  void onBackground() { std::cout << "App::onBackground()" << std::endl; }
  void onForeground() { std::cout << "App::onForeground()" << std::endl; }
  void onLimited() { std::cout << "App::onLimited()" << std::endl; }
  void onFull() { std::cout << "App::onFull()" << std::endl; }
}