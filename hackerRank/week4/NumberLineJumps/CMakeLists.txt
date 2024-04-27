cmake_minimum_required(VERSION 3.5)

project(NumberLineJumps LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(NumberLineJumps main.cpp)

include(GNUInstallDirs)
install(TARGETS NumberLineJumps
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
