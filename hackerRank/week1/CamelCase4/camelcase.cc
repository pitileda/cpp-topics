#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

void covertFirstChar(const char& ch, string& word) {
  switch (ch) {
    case 'M':
    case 'V':
      word[0] = tolower(word[0]);
      break;
    case 'C':
      word[0] = toupper(word[0]);
      break;
  }
};

string parseInputLine(const char& op, const char& lex, const string& str) {
  string word, out;
  switch (op) {
    case 'S': {
      for (char c : str) {
        if (isupper(c)) {
          out == "" ? out : out += ' ';
          out += tolower(c);
        } else {
          out += c;
        }
      }
      break;
    };
    case 'C': {
      istringstream iss(str);
      iss >> word;
      covertFirstChar(lex, word);
      out = word;
      while (iss >> word) {
        word[0] = toupper(word[0]);
        out += word;
      }
      if (lex == 'M') {
        out += "()";
      }
      break;
    }
  }
  return out;
};

int main() {
  /* Enter your code here. Read input from STDIN. Print output to STDOUT */
  while (!cin.eof()) {
    string line;
    getline(cin, line);
    if (cin.fail()) {
      break;
    }
    cout << parseInputLine(line[0], line[2], line.substr(4)) << std::endl;
  }
  return 0;
}
