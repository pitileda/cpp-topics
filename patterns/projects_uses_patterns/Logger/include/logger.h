#include <iostream>
#include <map>
#include <mutex>
#include <string>
namespace Logger {

enum class Level {
  Debug,
  Info,
  Warning,
  Error,
};

class Logger {
 private:
  Logger() = default;

 public:
  ~Logger() = default;
  Logger(const Logger&&);
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  static Logger& get();

  auto log(Level level, const std::string& msg) -> void;

  auto operator()(Level level, const std::string& msg) -> void;

 private:
  std::mutex mtx;
  std::map<Level, bool> allowed{
      {Level::Info, false},
      {Level::Debug, false},
      {Level::Warning, false},
      {Level::Error, true},
  };

  std::string formatMsg(Level level, const std::string& msg);
};

using enum Level;
}  // namespace Logger