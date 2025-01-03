#include <bits/stdc++.h>
#include <numeric>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);
vector<string> split(const string &);

/*
 * Complete the 'sansaXor' function below.
 *
 * The function is expected to return an INTEGER.
 * The function accepts INTEGER_ARRAY arr as parameter.
 */

int sansaXor(vector<int> arr) {
  auto getArrays = [](const vector<int> input) -> vector<vector<int>> {
    vector<vector<int>> output;
    unsigned int new_size = 2;
    while (new_size <= input.size()) {
      for (unsigned int i = 0; i < input.size(); ++i) {
        if (i + new_size > input.size()) {
          break;
        }
        vector<int> curr;
        curr.reserve(new_size);
        curr.push_back(input[i]);
        for (unsigned int j = i + 1; j < i + new_size; ++j) {
          curr.push_back(input[j]);
        }
        output.push_back(curr);
      }
      new_size++;
    }
    return output;
  };
  vector<vector<int>> sub_arr(getArrays(arr));
  int res = arr[0];
  for (unsigned int i = 1; i < arr.size(); ++i) {
    res = res ^ arr[i];
  }
  for (unsigned int i = 0; i < sub_arr.size(); ++i) {
    for (unsigned int j = 0; j < sub_arr[i].size(); ++j) {
      res = res ^ sub_arr[i][j];
    }
  }
  return res;
}

int fibonacciModified(int t1, int t2, int n) {
  auto fib = [](int n) {
    int a = 0;
    int b = 1;
    while (a < n) {
      int c = b;
      b = a + b * b;
      a = c;
    }
    return b;
  };

  return fib(t1 + n - 1);
}

string balancedSums(vector<int> arr) {
  unsigned int left = 0;
  unsigned int right = accumulate(arr.begin(), arr.end(), 0);
  for (auto i = 0; i < arr.size(); ++i) {
    right -= arr[i];
    if (left == right) {
      return string("YES");
    }
    left += arr[i];
  }
  return string("NO");
}

int superDigit(string n, int k) {
  if (n.size() == 1) {
    return (int)n[0];
  }
  int res = 0;
  for (auto times = 0; times < k; ++times) {
    for (auto i = 0; i < n.size(); ++i) {
      int curr = n[i] - '0';
      res += curr;
    }
  }
  while (res >= 10) {
    stringstream ss;
    ss << res;
    res = superDigit(ss.str(), 1);
  }
  return res;
}

long sumXor(long n) {
  auto countZeros = [](const long x) -> long {
    if (x == 0)
      return 1;
    long long input = x;
    long count{0};
    while (input > 0) {
      if (input % 2 == 0) {
        count++;
      }
      input /= 2;
    }
    return count;
  };
  return pow(2, countZeros(n));
}

int main() {
  long x = sumXor(8223372036854775807);
  superDigit(string("148"), 3);
  balancedSums({0, 0, 2, 0});
  fibonacciModified(0, 1, 5);
  int res = sansaXor({1, 2, 3});
  ofstream fout(getenv("OUTPUT_PATH"));

  string t_temp;
  getline(cin, t_temp);

  int t = stoi(ltrim(rtrim(t_temp)));

  for (int t_itr = 0; t_itr < t; t_itr++) {
    string n_temp;
    getline(cin, n_temp);

    int n = stoi(ltrim(rtrim(n_temp)));

    string arr_temp_temp;
    getline(cin, arr_temp_temp);

    vector<string> arr_temp = split(rtrim(arr_temp_temp));

    vector<int> arr(n);

    for (int i = 0; i < n; i++) {
      int arr_item = stoi(arr_temp[i]);

      arr[i] = arr_item;
    }

    int result = sansaXor(arr);

    fout << result << "\n";
  }

  fout.close();

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

vector<string> split(const string &str) {
  vector<string> tokens;

  string::size_type start = 0;
  string::size_type end = 0;

  while ((end = str.find(" ", start)) != string::npos) {
    tokens.push_back(str.substr(start, end - start));

    start = end + 1;
  }

  tokens.push_back(str.substr(start));

  return tokens;
}
