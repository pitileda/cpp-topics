include(${CMAKE_SOURCE_DIR}/cmake/sources.cmake)

# include_directories(${CMAKE_SOURCE_DIR}/destructor)
# include_directories(${googletest_SOURCE_DIR}/include)
# set(SOURCES ${CMAKE_SOURCE_DIR}/destructor/destructor.cc)
collect_sources(SOURCES ${CMAKE_SOURCE_DIR}/destructor/)
set(LIBRARIES
    gtest gmock gtest_main)

create_test("Destructor" ${SOURCES} ${LIBRARIES})