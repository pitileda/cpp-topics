#include <utility>
#include <tuple>
#include <string>
#include <iostream>
#include <memory>
#include <cxxabi.h>


// function to print the type of object
template <class T>
std::string
type_name()
{
    typedef typename std::remove_reference<T>::type TR;
    std::unique_ptr<char, void(*)(void*)> own
           (
#ifndef _MSC_VER
                abi::__cxa_demangle(typeid(TR).name(), nullptr,
                                           nullptr, nullptr),
#else
                nullptr,
#endif
                std::free
           );
    std::string r = own != nullptr ? own.get() : typeid(TR).name();
    if (std::is_const<TR>::value)
        r += " const";
    if (std::is_volatile<TR>::value)
        r += " volatile";
    if (std::is_lvalue_reference<T>::value)
        r += "&";
    else if (std::is_rvalue_reference<T>::value)
        r += "&&";
    return r;
}

int main(int argc, char const *argv[])
{
	std::pair p(0, 0ull);
	std::tuple t(0, 0.0, "0");

	const auto& [a, b, c] = t;

	std::cout 	<< type_name<decltype(a)>() << std::endl
				<< type_name<decltype(b)>() << std::endl
				<< type_name<decltype(c)>() ;
	return 0;
}