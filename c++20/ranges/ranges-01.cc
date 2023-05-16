#include <iostream>
#include <ranges>
#include <vector>

int main() {
  using std::views::filter,
  std::views::transform,
  std::views::reverse; // Some data for us to work on
  std::vector<int> numbers = { 6, 5, 4, 3, 2, 1 }; 

  // Lambda function that will provide filtering
  auto is_even = [](int n) { return n % 2 == 0; }; 

  // Process our dataset
  auto results = numbers
    | transform([](int n) { return n++; })
    | filter(is_even)
    | reverse; // Use lazy evaluation to print out the results
  for (auto v: results) {
    std::cout << v << " ";   // Output: 3 5 7
  }
}