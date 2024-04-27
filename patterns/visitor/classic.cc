#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
using namespace std;

struct Value;
struct AdditionExpression;
struct MultiplicationExpression;

struct ExpressionVisitor {
  virtual void visit(Value& v) = 0;
  virtual void visit(AdditionExpression& ae) = 0;
  virtual void visit(MultiplicationExpression& me) = 0;
};

struct Expression {
  virtual void accept(ExpressionVisitor& ev) = 0;
};

struct Value : Expression {
  int value;

  Value(int value) : value(value) {}
  void accept(ExpressionVisitor& ev) { ev.visit(*this); }
};

struct AdditionExpression : Expression {
  Expression &lhs, &rhs;

  AdditionExpression(Expression& lhs, Expression& rhs) : lhs(lhs), rhs(rhs) {}
  void accept(ExpressionVisitor& ev) { ev.visit(*this); }
};

struct MultiplicationExpression : Expression {
  Expression &lhs, &rhs;

  MultiplicationExpression(Expression& lhs, Expression& rhs)
      : lhs(lhs), rhs(rhs) {}
  void accept(ExpressionVisitor& ev) { ev.visit(*this); }
};

struct ExpressionPrinter : ExpressionVisitor {
  void visit(Value& v);
  void visit(AdditionExpression& ae);
  void visit(MultiplicationExpression& me);

  string str() const { return oss.str(); }
  std::ostringstream oss;
};

void ExpressionPrinter::visit(Value& v) { oss << v.value; }

void ExpressionPrinter::visit(AdditionExpression& ae) {
  oss << "(";
  ae.lhs.accept(*this);
  oss << "+";
  ae.rhs.accept(*this);
  oss << ")";
}

void ExpressionPrinter::visit(MultiplicationExpression& ae) {
  ae.lhs.accept(*this);
  oss << "*";
  ae.rhs.accept(*this);
}

int main(int argc, char const* argv[]) {
  Value v2{2};
  Value v3{3};
  AdditionExpression simple{v2, v3};
  ExpressionPrinter ep;
  ep.visit(simple);
  std::cout << ep.str();
  assert("(2+3)" == ep.str());
  return 0;
}