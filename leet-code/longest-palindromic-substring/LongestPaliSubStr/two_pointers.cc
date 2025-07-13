#include <cassert>
#include <cstdio>
#include <functional>
#include <string>

using namespace std;

class Solution {
 public:
  string longestPalindrome(string s) {
    if (s.size() < 2) return s;
    if (s.size() == 2) {
      return s[0] == s[1] ? s : string(1, s[0]);
    }
    string max_str;
    int left = -1, right = 1, i = 0;
    string curr_str;
    for (; i < s.size(); ++i, ++left, ++right) {
      if (left == -1 || right == s.size()) continue;
      curr_str += s[i];
      bool start{true};
      for (int ll = left, rr = right; ll >= 0 && rr < s.size(); --ll, ++rr) {
        if (s[ll] == s[rr]) {
          curr_str.insert(curr_str.begin(), s[ll]);
          curr_str.push_back(s[rr]);
        } else if (start && s[i] == s[rr]) {
          curr_str += s[rr];
          ll++;
        }
        curr_str.size() > max_str.size() ? max_str = curr_str : (max_str);
        start = false;
      }
      curr_str = "";
    }
    return max_str;
  }
};

auto testPalindrome = [](const string& input, const string& expected) {
  if (Solution sol; expected != sol.longestPalindrome(input)) {
    printf("actual: %s, expected: %s\n", sol.longestPalindrome(input).c_str(),
           expected.c_str());
  } else {
    printf("%s: Pass\n", input.c_str());
  }
};

int main() {
  testPalindrome("babad", "bab");
  testPalindrome("cbbd", "bb");
  testPalindrome("ac", "a");
  testPalindrome("ccd", "cc");
  return 0;
}