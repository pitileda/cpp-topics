cmake_minimum_required(VERSION 3.5)

project(Tasks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(async async.cc)
target_link_libraries(async ${CMAKE_THREAD_LIBS_INIT})

# include(GNUInstallDirs)
# install(TARGETS Tasks
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
