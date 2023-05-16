#include <memory>
#include <iostream>

class Single {
private:
	Single();
	~Single();
	Single(const Single& rhs) {
		single = rhs.single;
	}
	Single& operator=(const Single& rhs) {
		if (this != &rhs) {
			single = rhs.single;
		}

		return *this;
	}
	static Single* single;
public:
	static Single& getSingle();

	int get();
	int counter = 0;
};

Single::Single(){}
Single::~Single(){}

Single* Single::single = nullptr;

Single& Single::getSingle() {
	static Single sgl;
	single = &sgl;
	return *single;
}

int Single::get() {
	return 12;
}

int main(int argc, char const *argv[])
{
	Single* s = &Single::getSingle();
	s->get();
	Single::getSingle().counter++;
	std::cout << "s address is " << &Single::getSingle() << std::endl;
	std::cout << "s counter is " << Single::getSingle().counter << std::endl;

	return Single::getSingle().get();
}