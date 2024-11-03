#include "logger.h"

namespace Logger {

Logger::Logger(const Logger&&) { this->get(); }

Logger& Logger::get() {
  static Logger instance;
  return instance;
}

auto Logger::log(Level level, const std::string& msg) -> void {
  std::lock_guard<std::mutex> lck(mtx);
  std::cout << formatMsg(level, msg);
}

void Logger::operator()(Level level, const std::string& msg) {
  log(level, msg);
}

std::string Logger::formatMsg(Level level, const std::string& msg) {
  std::string out{""};
  if (allowed[level]) {
    switch (level) {
      case Level::Info: {
        out = "Info: " + msg;
        break;
      }
      case Level::Debug: {
        out = "Debug: " + msg;
        break;
      }
      case Level::Warning: {
        out = "Warning: " + msg;
        break;
      }
      case Level::Error: {
        out = "Error: " + msg;
        break;
      }
      default: {
        out = "default: " + msg;
        break;
      }
    }
    return out + '\n';
  }
  return out;
}
}  // namespace Logger