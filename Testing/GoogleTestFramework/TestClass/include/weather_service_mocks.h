#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <string>

#include "logger.h"
#include "weather_api_client.h"

class MockWeatherApiClient : public WeatherApiClient {
 public:
  MOCK_METHOD(std::string, fetchTemperature, (const std::string&), (override));
};

class MockLogger : public Logger {
 public:
  MOCK_METHOD(void, log, (const std::string&), (override));
};