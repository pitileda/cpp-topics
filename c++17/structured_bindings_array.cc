#include <cstdint>

int main(int argc, char const *argv[])
{
	long long theArray[4] = {1ull, 2ull, 3ull, 4ull};

	// auto [a, b, c] = theArray; will not compile, should decompose to 4 el
	auto [a, b, c, d] = theArray;
	return b;
}