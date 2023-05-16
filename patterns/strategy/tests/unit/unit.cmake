include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
  FetchContent_Populate(googletest)
  add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

add_executable(smoke_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/unit/src/smoke.cc)
target_include_directories(smoke_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(smoke_test gtest gmock gtest_main)
add_test(NAME gtest_smoke COMMAND smoke_test)