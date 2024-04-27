#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  int size;
  cin >> size;
  vector<int> v;
  v.reserve(size);
  for (size_t i = 0; i < 5; ++i) {
    cin >> size;
    v.push_back(size);
  }
  sort(v.begin(), v.end());
  cout << v[0];
  for (size_t i = 1; i < v.size(); i++) {
    cout << " " << v[i];
  }
  return 0;
}