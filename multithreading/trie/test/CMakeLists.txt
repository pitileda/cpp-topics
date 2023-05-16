# Add executable for benchmark test
add_executable(benchmark_test benchmark_test.cc)
target_compile_features(benchmark_test PRIVATE cxx_std_20)
target_include_directories(benchmark_test PRIVATE ${CMAKE_ROOT}/include)
target_include_directories(benchmark_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_sources(benchmark_test PRIVATE
    ${PROJECT_SOURCE_DIR}/src/trie.cc
    ${PROJECT_SOURCE_DIR}/src/trie_threadsafe.cc)
target_link_libraries(benchmark_test PRIVATE benchmark::benchmark)