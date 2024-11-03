cmake_minimum_required(VERSION 3.5)

project(Classic-AppsLevels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Classic-AppsLevels classic.cc)

include(GNUInstallDirs)
install(TARGETS Classic-AppsLevels
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
