#include <iostream>
#include <string>

using namespace std;

int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  string s1, s2;
  cin >> s1 >> s2;
  string res = s1;
  for (auto i = 0; i < s1.size(); ++i) {
    if (s1[i] == s2[i]) {
      res[i] = '0';
    } else {
      res[i] = '1';
    }
  }
  cout << res << endl;
  cout << stoi(string{res[res.size() - 1]});
  return 0;
}