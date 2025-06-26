cmake_minimum_required(VERSION 3.16)

project(Proxy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

add_executable(Proxy proxy_property.cpp)
add_executable(VirtualProxy virtual_proxy.cpp)

include(GNUInstallDirs)
install(TARGETS Proxy
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
