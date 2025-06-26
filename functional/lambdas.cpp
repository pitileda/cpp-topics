#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>

using namespace std;
struct FunctionAddValue {
    int value;
    int operator()(int x) {
        return x + this->value;
    }
    FunctionAddValue(int v) : value{v} {}
};

int call_add(std::function<void(fnv_analytics_event_info_t )> f) {
}


int main () {
    // Lambdas are annoymous functions which can be stored as values.
    auto print_lambda = [](){ cout << "Printing from a lambda" << endl; };
    // The body of a lambda is not run until it's called
    print_lambda();
    // And it may be called more than once
    print_lambda();
    print_lambda();

    // Lambdas can take arguments and return values
    auto add = [](int x, int y){ return x + y;};
    cout << "Two plus two is " << add(2, 2) << endl;

    // Lambdas can also capture variables from their environment
    int three = 3;
    auto add_three = [three](int x){ return three + x;};

    // Use = to copy all variables mentioned by value.
    int four = 4;
    auto add_four = [=](int x){ return four + x;};

    // Use & to copy all variables mentioned by reference. Use &var to copy by ref explicitly
    string msg {"Hello from another lambda"};
    auto send_msg = [&](){ cout << msg << endl;};
    auto send_msg2 = [&msg, four](){ cout << msg << four << endl;}; // msg by ref, four by value
    auto modify_four = [&four](int modify){four = modify;};

    // Lambdas basically desugar to callable structures.
    int seven = 7;
    // These are the same
    auto add_seven = [=](int x){return x + seven;};
    auto add_seven2 = FunctionAddValue{seven};
    cout << "Add seven to 4: " << add_seven(4) << endl;
    cout << "Add seven to 5: " << add_seven2(5) << endl;

    cout << call_add([](int x) {return x + 4;}, 6) << endl;
}
