#include <string>
#include <unordered_map>

bool anagram(std::string s1, std::string s2) {
  std::unordered_map<char, int> chars_count;
  for (char c : s1) {
    chars_count[c]++;
  }
  for (char c : s2) {
    chars_count[c]--;
    if (chars_count[c] == -1) return false;
  }
  for (auto [ch, count] : chars_count) {
    if (count > 0) {
      return false;
    }
  }
  return true;
}

int main() { return anagram(std::string{"cats"}, std::string{"tocs"}); }