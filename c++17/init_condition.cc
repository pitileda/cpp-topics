#include <cstdint>
#include <map>
#include <iostream>

int get(){
	return 12;
}

int main(int argc, char const *argv[])
{
	int i = 4;
	if (auto x = get(); x)
	{
		i = 4 * x;
	}

	switch(auto x = get(); x){
		case 1: {
			std::cout << "Hi\n";
			break;
		}
		default: {
			std::cout << x << ' ' << i << std::endl;
		}
	}
	return 0;
}