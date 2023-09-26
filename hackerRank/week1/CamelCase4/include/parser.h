#pragma once

#include <sstream>

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
  if (str.size() < 5) {
    return "";
  }

  string word, out, copy;
  copy = str.substr(4);
  switch (op) {
    case 'S': {
      for (char c : copy) {
        if (c == '(' || c == ')') {
          continue;
        }
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
      istringstream iss(copy);
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