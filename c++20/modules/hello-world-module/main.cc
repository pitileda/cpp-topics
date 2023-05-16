// compilation order matters now!!!
// g++ -fmodules-ts foo.cc main.cc - works
// g++ -fmodules-ts main.cc foo.cc - doesn't work
import foo;

int main(int argc, char const *argv[])
{
	foo f;
	f.helloworld();
	return 0;
}