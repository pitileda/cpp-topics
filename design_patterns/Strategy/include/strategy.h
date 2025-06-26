#pragma once
#include <iostream>

class PaymentStrategy {
public:
    virtual ~PaymentStrategy() = default;
    virtual void pay(double amount) const = 0;
};

class CreditCardPayment final : public PaymentStrategy {
public:
    explicit CreditCardPayment(const std::string& cardNumber) : cardNumber(cardNumber) {}
    void pay(double amount) const override {
        std::cout << "Paid " << amount << " using credit card " << cardNumber << std::endl;
    }
private:
    std::string cardNumber;
};

class PayPalPayment final : public PaymentStrategy {
public:
    explicit PayPalPayment(const std::string& email) : email(email) {}
    void pay(double amount) const override {
        std::cout << "Paid " << amount << " using PayPal account " << email << std::endl;
    }
private:
    std::string email;
};

class PaymentProcessor {
public:
    explicit PaymentProcessor(std::unique_ptr<PaymentStrategy> strategy): strategy_(std::move(strategy)) {}
    ~PaymentProcessor() {}

    void setStrategy(std::unique_ptr<PaymentStrategy> newStrategy) {
        strategy_ = std::move(newStrategy);
    }

    void processPayment(double amount) const {
        strategy_->pay(amount);
    }

private:
    std::unique_ptr<PaymentStrategy> strategy_;
};