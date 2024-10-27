cmake_minimum_required(VERSION 3.5)

project(Thread-safe-initialization LANGUAGES CXX)

find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(constexpr constexpr.cc)
target_link_libraries(constexpr ${CMAKE_THREAD_LIBS_INIT})

add_executable(call_once call_once.cc)
target_link_libraries(call_once ${CMAKE_THREAD_LIBS_INIT})

add_executable(deadlock deadlock.cc)
target_link_libraries(deadlock ${CMAKE_THREAD_LIBS_INIT})

add_executable(reader_writer reader_writer_lock.cc)
target_link_libraries(reader_writer ${CMAKE_THREAD_LIBS_INIT})

add_executable(thread_local thread_local.cc)
target_link_libraries(thread_local ${CMAKE_THREAD_LIBS_INIT})

add_executable(sender_receiver sender_receiver.cc)
target_link_libraries(sender_receiver ${CMAKE_THREAD_LIBS_INIT})

# include(GNUInstallDirs)
# install(TARGETS constexpr
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
