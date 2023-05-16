#include <set>
#include <cstdint>

using namespace std;

struct S {
	bool b;
	uint32_t u;

	bool operator < (S const & item) const{
		return item.u < u;
	}
};

int main(int argc, char const *argv[])
{
	set<S> setOfS{{true, 12}};

	//here we declare 2 vars
	auto [i, success] = setOfS.insert({false, 4});
	if (!success)
	{
		return 1;
	}

	return i->u;
}