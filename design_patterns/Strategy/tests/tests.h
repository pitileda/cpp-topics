#pragma once
#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "strategy.h"
#include <functional>

class MockPaymentStrategy : public  PaymentStrategy {
public:
    MOCK_METHOD(void, pay, (double amount), (const, override));
};

class PaymentProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        auto mock = std::make_unique<MockPaymentStrategy>();
        mockStrategy = mock.get();
        processor = std::make_unique<PaymentProcessor>(std::move(mock));
    }

    MockPaymentStrategy* mockStrategy;
    std::unique_ptr<PaymentProcessor> processor;
};

TEST_F(PaymentProcessorTest, ProcessesPaymentWithCurrentStrategy) {
    EXPECT_CALL(*mockStrategy, pay(100.0)).Times(1);
    processor->processPayment(100.0);
}

TEST_F(PaymentProcessorTest, CanChangeStrategy) {
    auto newMock = std::make_unique<MockPaymentStrategy>();
    auto* newMockPtr = newMock.get();

    EXPECT_CALL(*newMockPtr, pay(50.0))
        .Times(1);
    EXPECT_CALL(*mockStrategy, pay(::testing::_))
        .Times(0);

    processor->setStrategy(std::move(newMock));
    processor->processPayment(50.0);

    // Update our observer pointer
    mockStrategy = newMockPtr;
}