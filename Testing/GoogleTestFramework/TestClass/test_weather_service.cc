#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>

#include "weather_service.h"
#include "weather_service_mocks.h"

using ::testing::Return;

class WeatherServiceTest : public testing::Test {
 protected:
  std::shared_ptr<MockWeatherApiClient> api =
      std::make_shared<MockWeatherApiClient>();
  std::shared_ptr<MockLogger> logger = std::make_shared<MockLogger>();
  WeatherService service{api, logger};
};

class WeatherServiceTestParam
    : public testing::TestWithParam<std::pair<std::string, std::string>> {
 protected:
  std::shared_ptr<MockWeatherApiClient> api =
      std::make_shared<MockWeatherApiClient>();
  std::shared_ptr<MockLogger> logger = std::make_shared<MockLogger>();
  WeatherService service{api, logger};
};

TEST_F(WeatherServiceTest, Basic) {
  const std::string city{"London"};
  EXPECT_CALL(*api, fetchTemperature(city));
  service.getForecast(city);
}

TEST_P(WeatherServiceTestParam, Basic) {
  const auto& [city, expected] = GetParam();
  EXPECT_CALL(*api, fetchTemperature(city)).WillOnce(Return(expected));
  auto result = service.getForecast(city);
  EXPECT_EQ(result, expected);
}

INSTANTIATE_TEST_SUITE_P(Params, WeatherServiceTestParam,
                         ::testing::Values(std::make_pair("Berlin", "19"),
                                           std::make_pair("London", "18")));