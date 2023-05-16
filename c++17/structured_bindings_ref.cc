#include <cstdint>

int main(int argc, char const *argv[])
{
	long long theArray[4] = {1ull, 2ull, 3ull, 4ull};

	// get the ref-s 
	auto& [a, b, c, d] = theArray;
	b = 7ull;
	return b;
}