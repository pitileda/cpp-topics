#include <iostream>
#include <cmath>

int main() {
    double number1 = 3.14;
    double number2 = 5.0;
    double integerPart;

    // Check number1
    double fractionalPart1 = modf(number1, &integerPart);
    if (fractionalPart1 != 0.0) {
        std::cout << number1 << " has a fractional part." << std::endl;
    } else {
        std::cout << number1 << " does not have a fractional part." << std::endl;
    }

    // Check number2
    double fractionalPart2 = modf(number2, &integerPart);
    if (fractionalPart2 != 0.0) {
        std::cout << number2 << " has a fractional part." << std::endl;
    } else {
        std::cout << number2 << " does not have a fractional part." << std::endl;
    }

    return 0;
}
