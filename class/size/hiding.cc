#include <iostream>

class EmptyClassWithFunctions {
public: 
    virtual void display() {}
};

struct Ihor : virtual public EmptyClassWithFunctions {
    bool i = true;
    bool y = true;
    int x = 10;
};

struct Lena : virtual public EmptyClassWithFunctions {
    bool i = false;
    bool y = false;
    int x = 20;
};

struct Polina : public Ihor, public Lena {
    bool i = true;
    bool y = true;
    int x = 30;
};

int main() {
    Polina p;

    // Accessing Polina's own members
    std::cout << "Polina's i: " << p.i << "\n"; // Outputs: 1
    std::cout << "Polina's y: " << p.y << "\n"; // Outputs: 1
    std::cout << "Polina's x: " << p.x << "\n"; // Outputs: 30

    // Accessing Ihor's members
    std::cout << "Ihor's i: " << p.Ihor::i << "\n"; // Outputs: 1
    std::cout << "Ihor's y: " << p.Ihor::y << "\n"; // Outputs: 1
    std::cout << "Ihor's x: " << p.Ihor::x << "\n"; // Outputs: 10

    // Accessing Lena's members
    std::cout << "Lena's i: " << p.Lena::i << "\n"; // Outputs: 0
    std::cout << "Lena's y: " << p.Lena::y << "\n"; // Outputs: 0
    std::cout << "Lena's x: " << p.Lena::x << "\n"; // Outputs: 20

    return 0;
}
