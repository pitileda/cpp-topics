#pragma once
#include <gtest/gtest.h>

namespace dtor_test {

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

}