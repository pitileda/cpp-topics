#include <iostream>
#include <string>
#include <unordered_map>

char mostFrequentChar(std::string s) {
  std::unordered_map<char, int> frequency;
  int max_frequency = 0;
  for (char c : s) {
    frequency[c]++;
    if (frequency[c] > max_frequency) {
      max_frequency = frequency[c];
    }
  }

  char out = '\0';
  for (char c : s) {
    if (frequency[c] == max_frequency) {
      out = c;
      break;
    }
  }

  return out;
}

int main() {
  std::cout << mostFrequentChar(std::string{"mississippi"}) << std::endl;
  return 0;
}