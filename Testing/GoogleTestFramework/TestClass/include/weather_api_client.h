#pragma once

#include <string>

class WeatherApiClient {
 public:
  virtual ~WeatherApiClient() = default;
  virtual std::string fetchTemperature(const std::string& city) = 0;
};