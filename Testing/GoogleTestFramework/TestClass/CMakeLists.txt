cmake_minimum_required(VERSION 3.28 FATAL_ERROR)
project(TestClass VERSION 0.1.0 LANGUAGES CXX C)

enable_testing()

include(FetchContent)
FetchContent_Declare(
	googletest
	GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

add_executable(${PROJECT_NAME} test_class.cc)
target_link_libraries(${PROJECT_NAME} gtest gtest_main)
target_include_directories(${PROJECT_NAME} PRIVATE include)
target_compile_options(${PROJECT_NAME} PRIVATE
	-Werror -Wall -Wextra -Wpedantic -Wshadow -Wconversion)

add_executable(${PROJECT_NAME}-mock test_class_mocks.cc)
target_link_libraries(${PROJECT_NAME}-mock GTest::gtest GTest::gtest_main GTest::gmock)
target_include_directories(${PROJECT_NAME}-mock PRIVATE include)
target_compile_options(${PROJECT_NAME}-mock PRIVATE
	-Werror -Wall -Wextra -Wpedantic -Wshadow -Wconversion)

add_executable(WeatherServiceTest test_weather_service.cc)
target_link_libraries(WeatherServiceTest GTest::gtest GTest::gtest_main GTest::gmock gcov)
target_include_directories(WeatherServiceTest PRIVATE include)
target_compile_options(WeatherServiceTest PRIVATE
	-Werror -Wall -Wextra -Wpedantic -Wshadow -Wconversion)
target_compile_options(WeatherServiceTest PRIVATE --coverage -g)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
add_test(NAME ${PROJECT_NAME}-mock COMMAND ${PROJECT_NAME}-mock)
add_test(NAME WeatherServiceTest COMMAND WeatherServiceTest)
