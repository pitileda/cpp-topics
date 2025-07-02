#pragma once

#include <memory>
#include <string>

#include "logger.h"
#include "weather_api_client.h"

using ApiPtr = std::shared_ptr<WeatherApiClient>;
using LoggerPtr = std::shared_ptr<Logger>;

class WeatherService {
 public:
  explicit WeatherService(ApiPtr api, LoggerPtr logger)
      : api_(api), logger_(logger) {}

  std::string getForecast(const std::string& city) {
    logger_->log("Getting forecast for " + city);
    auto res = api_->fetchTemperature(city);
    logger_->log("Received forecast " + res);
    return res;
  }

 private:
  std::shared_ptr<WeatherApiClient> api_;
  std::shared_ptr<Logger> logger_;
};