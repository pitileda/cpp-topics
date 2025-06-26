#pragma once
#include "gtest/gtest.h"
#include "sentence.h"

class HappyPathTest : public  ::testing::Test {
protected:
    HappyPathTest() :
    empty(""),
    simple("hello ihor"),
    multi_space("hello   ihor"),
    end_space("hello sds "),
    single_word("hello")
    {}

    std::string empty;
    std::string simple;
    std::string multi_space;
    std::string end_space;
    std::string single_word;
};

TEST_F(HappyPathTest, CtorEmptyString) {
    Sentence em(empty);
    EXPECT_EQ(0, em[0].start);
    EXPECT_EQ(0, em[0].end);
}

TEST_F(HappyPathTest, CtorHappyString) {
    Sentence em(simple);
    EXPECT_EQ(0, em[0].start);
    EXPECT_EQ(4, em[0].end);
}