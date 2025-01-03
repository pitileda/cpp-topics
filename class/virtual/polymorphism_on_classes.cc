// https://www.fluentcpp.com/2020/05/15/runtime-polymorphism-without-virtual-functions/
// but I modified it to use with non static member functions

#include <algorithm>
#include <cstdio>
#include <functional>
#include <iterator>
#include <vector>

struct Input {
  int value;
};

struct Output {
  int value;
};

struct SmallCalculator {
  bool is_applicable(const Input& in) { return in.value <= 10; }

  Output compute(const Input& in) {
    printf("SmallCalculator->Input value: %d, Output value: %d", in.value,
           in.value * 2);
    return Output{in.value * 2};
  }
};

struct BigCalculator {
  bool is_applicable(const Input& in) { return in.value > 10; }

  Output compute(const Input& in) {
    printf("BigCalculator->Input value: %d, Output value: %d", in.value,
           in.value * 5);
    return Output{in.value * 5};
  }
};

struct Calculator {
  std::function<bool(const Input&)> is_applicable;
  std::function<Output(const Input&)> compute;

  template <typename CalculatorType>
  static Calculator create_from(CalculatorType& obj) {
    return Calculator{
        [&obj](const Input& in) -> bool { return obj.is_applicable(in); },
        [&obj](const Input& in) -> Output { return obj.compute(in); }};
  }
};

int main(int argc, char const* argv[]) {
  SmallCalculator smallCalculator;
  BigCalculator bigCalculator;
  std::vector calculators{Calculator::create_from(smallCalculator),
                          Calculator::create_from(bigCalculator)};
  Input value{3};
  auto calculator_it = std::find_if(
      std::begin(calculators), std::end(calculators),
      [&value](auto& calculator) { return calculator.is_applicable(value); });
  auto calc = *calculator_it;
  calc.compute(value);
  return 0;
}
