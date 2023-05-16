#include <iostream>
#include <vector>

// simple usage
template <typename ...Argc>
auto sum(Argc ...argc){
	return (argc + ... + 0);
}

// with cout and forwarding references
template<typename ...Argc>
void printf(Argc&& ...argc){
	(std::cout << ...  << std::forward<Argc>(argc)) << '\n';
}

// using comma operator
template<typename T, typename ...Argc>
void fillVec(std::vector<T>& v, Argc&& ...argc){
	(v.push_back(argc), ...);
}

// with cout and comma operator to have spaces
template<typename ...Argc>
void printfs(Argc&& ...argc){
	const char sep = ' ';
	((std::cout << std::forward<Argc>(argc) << sep), ...);
	std::cout << std::endl;
}

int main(int argc, char const *argv[])
{
	printf(12, 'c', 12.4);
	printfs(12, 'c', 12.4);
	std::vector<char> cv;
	fillVec(cv, 'c', 'b', 'f', 'g');
	printf(cv[0], cv[1], cv[2], cv[3]);
	return sum(1,2,4,5);
}