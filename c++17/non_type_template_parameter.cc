#include <string>

// prior c++17
// template <typename Type, Type value> constexpr Type TConstant = value;

// after c++17
template <auto value> constexpr auto TConstant = value;

int main(int argc, char const *argv[])
{
	// prior c++17
	// constexpr auto const MySuperConst = TConstant<int, 100>;
	constexpr auto const MySuperConst = TConstant<100>;
	return MySuperConst;
}