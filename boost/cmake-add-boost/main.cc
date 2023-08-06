#include "mylib.h"
#include <boost/filesystem.hpp>
#include <iostream>
#include <memory>

using namespace boost::filesystem;

int main(int argc, char const *argv[])
{
	if (argc < 2)
	{
		std::cout << "use with an argument\n";
		return 1;
	}

	std::cout << file_size(argv[1]) << '\n';
	std::cout << "Привет!" << std::endl;

	std::shared_ptr<My> libObj = std::make_shared<Derived>();
	return 0;
}