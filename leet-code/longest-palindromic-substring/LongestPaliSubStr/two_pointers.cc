#include <cassert>
#include <cstdio>
#include <functional>
#include <string>

using namespace std;

class Solution {
public:
  string longestPalindrome(string s) {
    if (s.empty()) {
      return "";
    }

    int start = 0;
    int end = 0;

    for (int i = 0; i < s.length(); i++) {
      int odd = expandAroundCenter(s, i, i);
      int even = expandAroundCenter(s, i, i + 1);
      int max_len = max(odd, even);

      if (max_len > end - start) {
        start = i - (max_len - 1) / 2;
        end = i + max_len / 2;
      }
    }

    return s.substr(start, end - start + 1);
  }

private:
  int expandAroundCenter(string s, int left, int right) {
    while (left >= 0 && right < s.length() && s[left] == s[right]) {
      left--;
      right++;
    }
    return right - left - 1;
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
  testPalindrome("aaaa", "aaaa");
  return 0;
}