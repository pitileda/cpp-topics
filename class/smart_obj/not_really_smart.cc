#include <iostream>
#include <string>
#include <type_traits>

template<typename T>
class SmartObject {
private:
    T value;

public:
    // Constructor that accepts any numeric type or string
    SmartObject(T val) : value(val) {
        static_assert(std::is_arithmetic<T>::value || std::is_same<T, std::string>::value,
                      "SmartObject can only be instantiated with numeric types or std::string");
    }

    // Function to set the value
    void setValue(T val) {
        value = val;
    }

    // Function to get the value
    T getValue() const {
        return value;
    }

    // Function to print the value
    void print() const {
        std::cout << "SmartObject value: " << value << std::endl;
    }

    // Function to add another SmartObject of the same type
    SmartObject<T> operator+(const SmartObject<T>& other) const {
        return SmartObject<T>(value + other.value);
    }

    // Function to subtract another SmartObject of the same type
    SmartObject<T> operator-(const SmartObject<T>& other) const {
        return SmartObject<T>(value - other.value);
    }

    // Function to multiply another SmartObject of the same type
    SmartObject<T> operator*(const SmartObject<T>& other) const {
        return SmartObject<T>(value * other.value);
    }

    // Function to divide another SmartObject of the same type
    SmartObject<T> operator/(const SmartObject<T>& other) const {
        static_assert(!std::is_same<T, std::string>::value,
                      "Division is not supported for std::string type");
        return SmartObject<T>(value / other.value);
    }
};

int main() {
    // Examples with different types
    SmartObject<int> obj1(10);
    SmartObject<double> obj2(20.5);
    SmartObject<std::string> obj3("Hello");

    obj1.print();  // Outputs: SmartObject value: 10
    obj2.print();  // Outputs: SmartObject value: 20.5
    obj3.print();  // Outputs: SmartObject value: Hello

    // Example of arithmetic operations
    SmartObject<int> obj4 = obj1 + SmartObject<int>(5);
    obj4.print();  // Outputs: SmartObject value: 15

    SmartObject<double> obj5 = obj2 - SmartObject<double>(5.5);
    obj5.print();  // Outputs: SmartObject value: 15

    // String concatenation
    SmartObject<std::string> obj6 = obj3 + SmartObject<std::string>(" World");
    obj6.print();  // Outputs: SmartObject value: Hello World

    return 0;
}
