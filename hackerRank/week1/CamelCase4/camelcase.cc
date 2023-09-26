#include <iostream>
#include <string>

#include "parser.h"
using namespace std;

int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  while (!cin.eof()) {
    string line;
    getline(cin, line);
    if (cin.fail()) {
      break;
    }
    if (line.size() < 5) {
      break;
    }
    cout << parseInputLine(line[0], line[2], line) << std::endl;
  }
  return 0;
}
