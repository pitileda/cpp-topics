#include "dtor.h"
#include <memory>

namespace dtor_test {

TEST(Destructor, IsCalledForBothBaseAndDerivedWithVirtual) {
    {
        Foo* foo = new Bar();
        delete foo;
    }
    EXPECT_TRUE(Foo::destructor_called_);
    EXPECT_TRUE(Bar::destructor_called_);
}

TEST(Destructor, IsCalledForBothBaseAndDerivedWOVirtual){
    {
        std::shared_ptr<Base> base = std::make_shared<Derived>();
    }

    EXPECT_TRUE(Base::destructor_called_);
    EXPECT_TRUE(Derived::destructor_called_);
}

}