#include <iostream>
#include "mylib.h"

My::My()
{
	std::cout << "ctor My\n";
}

My::~My()
{
	std::cout << "dtor My\n";
}

Derived::Derived()
{
	std::cout << "ctor Derived" << std::endl;
}

Derived::~Derived()
{
	std::cout << "dtor Derived" << std::endl;
}

int calculate(){
	return 12;
}