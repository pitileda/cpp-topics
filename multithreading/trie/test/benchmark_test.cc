#include <benchmark/benchmark.h>

#include <string_view>
#include <thread>
#include <vector>

#include "trie.h"
#include "trie_threadsafe.h"

static void benchmark_insert_single_word(benchmark::State& state) {
  const std::string word = "Hello";
  ikov::Trie trie;
  for (auto _ : state) {
    trie.insert(word);
  }
}

static void benchmark_insert_multiple_words(benchmark::State& state) {
  ikov::Trie trie;
  const int num_words = state.range(0);
  std::vector<std::string> words(num_words);
  for (size_t i = 0; i < num_words; i++) {
    words[i] = "word_" + std::to_string(i);
  }

  for (auto _ : state) {
    for (const auto& word : words) {
      trie.insert(word);
    }
  }
}

static void benchmark_insert_InsertThreaded(benchmark::State& state) {
  const int num_threads = state.range(0);
  const int num_words = 1 << 20;
  const int words_per_thread = num_words / num_threads;

  ikov::TrieThreadSafe t;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back([&t, i, words_per_thread] {
      for (int j = 0; j < words_per_thread; ++j) {
        t.insert("word_" + std::to_string(i * words_per_thread + j));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (auto _ : state) {
    state.PauseTiming();
    t.clear();
    state.ResumeTiming();
  }
}

BENCHMARK(benchmark_insert_single_word);
BENCHMARK(benchmark_insert_multiple_words)->Range(1, 1 << 20);
BENCHMARK(benchmark_insert_InsertThreaded)->Range(1, 32);

// Define main function to run the benchmark tests
int main(int argc, char** argv) {
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
