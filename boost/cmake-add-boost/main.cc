#include <boost/filesystem.hpp>
#include <iostream>

using namespace boost::filesystem;

int main(int argc, char const* argv[])
{
    if (argc < 2) {
        std::cout << "use with an argument\n";
        return 1;
    }
    std::cout << file_size(argv[1]) << '\n';

    return 0;
}