cmake_minimum_required(VERSION 3.5)

project(PickingNumbers LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(PickingNumbers main.cpp)

include(GNUInstallDirs)
install(TARGETS PickingNumbers
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
