#include <gtest/gtest.h>
#include <memory>

namespace {

class Foo {
    public:
    virtual ~Foo(){
        Foo:destructor_called_ =  true;
    }
    static bool destructor_called_;
};

class Bar : public Foo
{
    public:
    ~Bar(){
        Bar::destructor_called_ = true;
    }
    static bool destructor_called_;
};
bool Foo::destructor_called_ = false;
bool Bar::destructor_called_ = false;

class Base {
    public:
    static bool destructor_called_;
    ~Base(){
        Base::destructor_called_ = true;
    }
};

class Derived : public Base {
    public:
    static bool destructor_called_;
    ~Derived(){
        Derived::destructor_called_ = true;
    }
};
bool Base::destructor_called_ = false;
bool Derived::destructor_called_ = false;

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