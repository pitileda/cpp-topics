#include <iostream>
#include <ostream>
#include <sstream>
#include <utility>
#include <vector>

enum class TokenType {
  plus, minus, lparen, rparen, integer, multiply, divide
};

class Token {
public:
  Token(const TokenType Type, std::string Name) : type(Type), text(std::move(Name)) {}
  friend std::ostream &operator<<(std::ostream &Os, const Token &Obj) {
    return Os << "`" << Obj.text << "`";
  }

private:
  TokenType type;
  std::string text;
};

std::vector<Token> lex(const std::string& input) {
  std::vector<Token> result;
  for (auto i = 0; i < input.size(); ++i) {
    switch (input[i]) {
      case '(' : {
        result.emplace_back(TokenType::lparen, "(");
        break;
      }
      case ')' : {
        result.emplace_back(TokenType::rparen, ")");
        break;
      }
      case '+' : {
        result.emplace_back(TokenType::plus, "+");
        break;
      }
      case '-' : {
        result.emplace_back(TokenType::minus, "-");
        break;
      }
      case '*' : {
        result.emplace_back(TokenType::multiply, "*");
        break;
      }
      case '/' : case ':' : {
        result.emplace_back(TokenType::divide, ":");
        break;
      }
        default: {
        std::stringstream buffer;
        buffer << input[i];
        for (auto j = i+1; j < input.size(); ++j) {
          if (isdigit(input[j])) {
            buffer << input[j];
            if (i + 1 == input.size() - 1) {
              result.emplace_back(TokenType::integer, buffer.str());
            }
            ++i;
          } else {
            result.emplace_back(TokenType::integer, buffer.str());
            break;
          }
        }
      }
    }
  }
  return  result;
}

int main() {
  std::string input{"(12-1)+(145-18)*13"};
  auto res= lex(input);
  for (const auto& item: res) {
    std::cout << item << " ";
  }
  return 0;
}
