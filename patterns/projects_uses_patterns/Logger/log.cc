#include "logger.h"

using enum Logger::Level;

int main() {
  Logger::Logger::get().log(Info, "Hello");
  Logger::Logger logger = std::move(Logger::Logger::get());
  logger(Error, "Byu!");
  return 0;
}