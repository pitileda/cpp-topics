#include <bits/stdc++.h>

#include <ostream>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);

/*
 * Complete the 'separateNumbers' function below.
 *
 * The function accepts STRING s as parameter.
 */

void separateNumbers(string s) {
  string curr_str{""};
  size_t current{0};
  int prev{0};
  bool yes = false;
  size_t start, end;
  start = end = s.size() - 1;
  while (start >= 0 && end >= 0) {
    if (!yes) {
      if (s[start] == '0') {
        --start;
        continue;
      }
      curr_str = s.substr(start, end - start + 1);
      current = stoi(curr_str);
    }
    prev = current - 1;
    if (prev < 0) {
      yes = false;
      continue;
    }
    string prev_str = to_string(prev);
    // check if previous chars are next_str
    if ((int)(start - prev_str.size()) < 0) {
      yes = false;
      break;
    }
    size_t prev_end = start - 1;
    size_t prev_start = start - prev_str.size();
    cout << "prev_start: " << prev_start << ".." << std::flush;
    if (s.substr(prev_start, (prev_end - prev_start) + 1) == prev_str) {
      current = prev;
      yes = true;
      start = prev_start;
      end = prev_end;
      if (start == 0) {
        break;
      }
    }
    // previous string is not our
    else {
      --start;
      yes = false;
    }
  }
  if (yes) {
    cout << "YES " << prev;
  } else {
    cout << "No";
  }
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
