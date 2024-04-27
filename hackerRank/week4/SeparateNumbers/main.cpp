#include <bits/stdc++.h>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);

/*
 * Complete the 'separateNumbers' function below.
 *
 * The function accepts STRING s as parameter.
 */

void separateNumbers(string s) {
  if (s.size() < 2) {
    cout << "NO\n";
    return;
  }
  auto str_to_number = [&s](const size_t &start, const size_t &end) {
    return s.substr(start, end - start);
  };
  int first = -1;
  size_t start = 0;
  size_t i = 1;
  while (i < s.size()) {
    string curr = str_to_number(start, i);
    int curr_num = stoi(curr);
    int next_num = curr_num + 1;
    stringstream ss;
    ss << next_num;
    string next = ss.str();
    if (s.substr(i, next.size()) == next) {
      if (first == -1) {
        first = i;
      }
      start = i;
      i += next.size();
      continue;
    }
    if (first == -1) {
      i++;
      continue;
    }
    cout << "NO\n";
    return;
  }
  if (first != -1) {
    cout << "YES " << s.substr(0, first) << endl;
  }
  cout << "No\n";
}

int main() {
  string q_temp;
  getline(cin, q_temp);

  int q = stoi(ltrim(rtrim(q_temp)));

  for (int q_itr = 0; q_itr < q; q_itr++) {
    string s;
    getline(cin, s);

    separateNumbers(s);
  }

  return 0;
}

string ltrim(const string &str) {
  string s(str);

  s.erase(s.begin(),
          find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace))));

  return s;
}

string rtrim(const string &str) {
  string s(str);

  s.erase(
      find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(),
      s.end());

  return s;
}
