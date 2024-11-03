cmake_minimum_required(VERSION 3.5)

project(Producer-Consumer-Problem LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(Producer-Consumer-Problem main.cpp)
target_link_libraries(Producer-Consumer-Problem ${CMAKE_THREAD_LIBS_INIT})

add_executable(gpt gpt.cc)
target_link_libraries(gpt ${CMAKE_THREAD_LIBS_INIT})

# include(GNUInstallDirs)
# install(TARGETS Producer-Consumer-Problem
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
