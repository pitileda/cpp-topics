cmake_minimum_required(VERSION 3.5)

project(Task1 LANGUAGES CXX)

add_executable(Task1 main.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE header)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Werror -Wpedantic)

include(GNUInstallDirs)
install(TARGETS Task1
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
