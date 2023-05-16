// https://stackoverflow.com/questions/1632600/memory-layout-c-objects
// https://eli.thegreenplace.net/2012/12/17/dumping-a-c-objects-memory-layout-with-clang/
// clang -cc1 -fdump-record-layouts <source file name>
// g++ -fdump-lang-class -c struct-size.cc -> g++8+
// g++ -fdump-class-hierarchy -c struct-size.cc -> prio g++8

// #include <iostream>

// using namespace std;

//Empty class
class EmptyClass{

};

//Empty class that contains only function
class EmptyClassWithFunctions{
public: 
	virtual void display(){
	}
};

struct Ihor : virtual public EmptyClassWithFunctions {
	bool i;
	bool y;
	int x;
};

struct Lena : virtual public EmptyClassWithFunctions {
	bool i;
	bool y;
	int x;
};

struct Polina : public Ihor, public Lena {
	bool i;
	bool y;
	int x;
};

int main(int argc, char const *argv[])
{
	// cout << sizeof(EmptyClassWithFunctions) ;
	// cout << sizeof(EmptyClass) ;
	// cout << sizeof(Ihor) ;

	// return sizeof(EmptyClassWithFunctions) ;
	// return sizeof(Ihor);
	return sizeof(Polina);
}