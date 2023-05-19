#include <iostream>
#include <string>

// in c++ ctor can be templated even if exact class is not

//helpers to work with a type as values

template<class...>
struct types{
	using type=types;
};

template<class T>
struct tag{
	using type=T;
};

// template<class Tag> using type_t=typename Tag::type;


// class using helper
class A
{
public:
	template<class T>
	A(tag<T>){}

	template<class T, class U, class V>
	A(types<T, U, V>){}
};

int main(int argc, char const *argv[])
{
	A a(tag<int>{});
	auto aa = A(types<int, double, std::string>{});
	return 0;
}