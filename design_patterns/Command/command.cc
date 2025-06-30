#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

class BankAccount {
public:
  BankAccount(int64_t Amount, int64_t OverdraftLimit) : amount(Amount), overdraft_limit(OverdraftLimit) {}

  int64_t deposit(int64_t value) {
    amount += value;
    return amount;
  }

  int64_t withdraw(int64_t value) {
    if (amount - value >= overdraft_limit) {
      amount -= value;
    }
    return amount;
  }
  friend std::ostream &operator<<(std::ostream &Os, const BankAccount &Obj) { return Os << "amount: " << Obj.amount; }

private:
  int64_t amount{0};
  int64_t overdraft_limit{-500};
};

class Command {
public:
  virtual ~Command() = default;
  virtual void call(const int64_t &value) = 0;
};

enum class Action { deposit, withdraw };

class BankAccountCommand final : public Command {
public:
  BankAccountCommand(BankAccount &BankAccount, const Action &action) : bank_account(BankAccount), action(action) {}
  void call(const int64_t &value) override {
    switch (action) {
      case Action::deposit:
        bank_account.deposit(value);
        break;
      case Action::withdraw:
        bank_account.withdraw(value);
        break;
    }
  }

private:
  BankAccount &bank_account;
  Action action;
};

int main() {
  BankAccount ba{100, -200};
  std::vector<std::unique_ptr<Command>> commands;
  commands.push_back(std::make_unique<BankAccountCommand>(ba, Action::deposit));
  commands.push_back(std::make_unique<BankAccountCommand>(ba, Action::withdraw));
  auto deposit = std::move(commands[0]);
  auto withdraw = std::move(commands[1]);
  // commands[0]->call(100);
  // commands[0]->call(100);
  // commands[1]->call(400);
  deposit->call(200);
  withdraw->call(700);
  std::cout << ba << std::endl;
  return 0;
}
