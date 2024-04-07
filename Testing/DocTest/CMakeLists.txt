cmake_minimum_required(VERSION 3.27)
project(DocTest LANGUAGES CXX)

find_package(doctest REQUIRED)

add_executable(${PROJECT_NAME} main.cc)
target_include_directories(${PROJECT_NAME} PRIVATE ${DOCTEST_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} doctest::doctest)
