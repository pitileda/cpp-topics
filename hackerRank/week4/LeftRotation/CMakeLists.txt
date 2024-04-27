cmake_minimum_required(VERSION 3.5)

project(LeftRotation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(LeftRotation main.cpp)

include(GNUInstallDirs)
install(TARGETS LeftRotation
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
