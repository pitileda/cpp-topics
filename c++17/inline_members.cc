#include <cstdint>
#include <iostream>

struct Ihor {
	inline static int i = 12;
};

int main(int argc, char const *argv[])
{
	return Ihor::i;
}