#include <iostream>
#include <set>
#include <string>
#include <string_view>

using namespace std;

class Solution {
 private:
  struct PalindromCmp {
    bool operator()(const string& lhs, const string& rhs) const {
      if (lhs == rhs) {
        return false;
      } else {
        return lhs.size() > rhs.size();
      }
    }
  };
  set<string, PalindromCmp> palindroms;

  bool isPalindrom(const string_view& sv, uint32_t s_pos, uint32_t e_pos) {
    // it is guarateed that s_pos and e_pos are equal
    bool result = true;
    for (int current = 0; current < (e_pos - s_pos + 1) / 2; current++) {
      if (sv[s_pos + current] != sv[e_pos - current]) {
        result = false;
        break;
      }
    }
    return result;
  }

 public:
  string longestPalindrome(string s) {
    if (s.size() == 1) {
      return s;
    }
    if (s.size() == 0) {
      return "";
    }
    string_view sv(s);
    // go trough all chars
    for (uint32_t current = 0; current < s.size(); current++) {
      // for current char iterate next till the same char
      for (uint32_t next = current + 1; next < s.size(); next++) {
        // if new char is the same then check that current substring is
        // palindrom
        if (sv[current] == sv[next] and isPalindrom(sv, current, next)) {
          palindroms.insert(string(sv.substr(current, next - current + 1)));
        }
      }
    }

    if (palindroms.size() > 0) {
      return *palindroms.begin();
    }
    return string(1, s[0]);
  }

  void reset() {
    palindroms.clear();
  }
};

int main() {
  Solution s;
  cout << s.longestPalindrome("babad") << " --> "
       << "bab, aba" << endl;
  s.reset();
  cout << s.longestPalindrome("babab") << " --> "
       << "babab" << endl;
  s.reset();
  cout << s.longestPalindrome("badad") << " --> "
       << "ada, dad" << endl;
  s.reset();
  cout << s.longestPalindrome("baqas") << " --> "
       << "aqa" << endl;
  s.reset();
  cout << s.longestPalindrome("baqes") << " --> "
       << "b" << endl;
  s.reset();
  cout << s.longestPalindrome("cbbd") << " --> "
       << "bb" << endl;
  s.reset();
  cout << s.longestPalindrome("") << " --> "
       << "" << endl;
  s.reset();
  cout << s.longestPalindrome("a") << " --> "
       << "a" << endl;
  s.reset();
  cout << s.longestPalindrome("bb") << " --> "
       << "bb" << endl;
  s.reset();
  cout << s.longestPalindrome("aacabdkacaa") << " --> "
       << "aca" << endl;
  s.reset();
  cout << s.longestPalindrome("abcdbbfcba") << " --> "
       << "bb" << endl;
  s.reset();
  cout << s.longestPalindrome(
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
              "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") << " --> " << endl;
  return 0;
}
