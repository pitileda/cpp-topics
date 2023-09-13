cmake_minimum_required(VERSION 3.14)
project(BoostExample)

find_package(Boost 1.70.0 REQUIRED system filesystem)

add_executable(boost_example main.cc)
target_link_libraries(boost_example PRIVATE Boost::headers Boost::system Boost::filesystem)