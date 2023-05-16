#include <iostream>
#include <memory>
#include "node.h"

int main(int, char**) {
    std::cout << "Hello, world!\n";
    std::shared_ptr<IMethod> mth = std::make_shared<PictureMethod>();
    Picture pic(mth);
    pic.ScaleToFit();
}
