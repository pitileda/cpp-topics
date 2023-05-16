#include <cstdint>
#include <map>
#include <iostream>

int main(int argc, char const *argv[])
{
	std::map<uint32_t, char> m{{1, 'c'}, {2, 'b'}};
	for (const auto& [k,v]: m)
	{
		std::cout << k << v << std::endl;
	}
	return 0;
}