/*
Use tag dispatching to provide additional info on behavior
Use enums           to provide additional info on data
*/

#include <iostream>

class MyClass
{
public:
	static struct ThisWay{} thisWay;
	static struct ThatWay{} thatWay;
	explicit MyClass(ThisWay);
	explicit MyClass(ThatWay);
};

MyClass::MyClass(ThisWay) {
	std::cout << "Was constructed ThisWay" << std::endl;
}

MyClass::MyClass(ThatWay) {
	std::cout << "Was constructed ThatWay" << std::endl;
}

int main(int argc, char const *argv[])
{
	MyClass thisWay(MyClass::thisWay);
	MyClass thatWay(MyClass::thatWay);
	return 0;
}