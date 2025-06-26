cmake_minimum_required(VERSION 3.31)
project(SharedPointer)

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
        boost
        GIT_REPOSITORY https://github.com/boostorg/boost.git
        GIT_TAG boost-1.88.0
        GIT_SUBMODULES_RECURSE ON
)

set(BOOST_ENABLE_CMAKE ON CACHE BOOL "Enable Cmake in boost")
set(BUILD_SHARED_LIBS OFF)

FetchContent_MakeAvailable(boost)

add_executable(SharedPointer shared_tests.cpp)
target_include_directories(SharedPointer PRIVATE shared_include)
target_link_libraries(SharedPointer PRIVATE Boost::unit_test_framework)

add_executable(UniquePointer unique_tests.cpp)
target_include_directories(UniquePointer PRIVATE unique_include)
target_link_libraries(UniquePointer PRIVATE Boost::unit_test_framework)

enable_testing()
add_test(NAME SharedPointerTests COMMAND SharedPointer)

install(TARGETS SharedPointer)
