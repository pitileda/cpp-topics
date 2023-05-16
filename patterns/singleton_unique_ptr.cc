#include <memory>
#include <iostream>

class Single {
private:
	Single();
	static std::unique_ptr<Single> single;
public:
	Single(const Single&) = delete;
	Single& operator=(const Single&) = delete;
	Single(const Single&& rhs);
	Single& operator=(const Single&& rhs);
	static Single& getSingle();
	~Single();

	int get();
	int counter = 0;
};

Single::Single(){}
Single::~Single(){}

Single::Single(const Single&& rhs) {
	this->single = std::move(rhs.single);
};

Single& Single::operator=(const Single&& rhs) {
	if (this != &rhs) {
		return *(rhs.single);
	}
	return *this;
};

Single& Single::getSingle() {
	if (single == nullptr) {
		single.reset(new Single);
	}
	return *single;
}

std::unique_ptr<Single> Single::single = nullptr;

int Single::get() {
	return 12;
}

int main(int argc, char const *argv[])
{
	Single a = std::move(Single::getSingle());
	Single b = std::move(a);
	return a.get();
}