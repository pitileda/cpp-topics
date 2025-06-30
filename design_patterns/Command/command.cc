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

  std::pair<int64_t, bool> withdraw(int64_t value) {
    if (amount - value >= overdraft_limit) {
      amount -= value;
      return {amount, true};
    }
    return {amount, false};
  }
  friend std::ostream &operator<<(std::ostream &Os, const BankAccount &Obj) { return Os << "amount: " << Obj.amount; }

private:
  int64_t amount{0};
  int64_t overdraft_limit{-500};
};

class Command {
protected:
  bool succeeded = false;
public:
  virtual ~Command() = default;
  virtual void call(const int64_t &value) = 0;
  virtual void call() = 0;
  virtual void undo(const int64_t &value) = 0;
  virtual void undo() = 0;
};

enum class Action { deposit, withdraw };

class BankAccountCommand final : public Command {
public:
  BankAccountCommand(BankAccount &BankAccount, Action Action, int64_t Amount) :
      bank_account(BankAccount), action(Action), amount(Amount) {
    succeeded = false;
  }
  BankAccountCommand(BankAccount &BankAccount, const Action &action) : bank_account(BankAccount), action(action) {
    succeeded = false;
  }
  void call(const int64_t &value) override {
    switch (action) {
      case Action::deposit: {
        bank_account.deposit(value);
        succeeded = true;
        break;
      }
      case Action::withdraw: {
        auto [new_value,succeeded] = bank_account.withdraw(value);
        break;
      }
    }
  }

  void call() override {
    call(amount);
  }

  void undo(const int64_t &value) override {
    if (!succeeded) {
      return;
    }
    switch (action) {
      case Action::deposit : {
        auto [new_value, succeeded] = bank_account.withdraw(value);
        break;
      }
      case Action::withdraw : {
        bank_account.deposit(value);
        succeeded = true;
        break;
      }
    }
  }

  void undo() override {
    undo(amount);
  }

private:
  BankAccount &bank_account;
  Action action;
  int64_t amount{0};
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
  deposit->call(200); //300
  withdraw->call(700); // 300
  withdraw->undo(700); // 300
  deposit->call(200); // 500
  deposit->call(300); // 800
  deposit->undo(400); // 400

  auto deposit100 = std::make_unique<BankAccountCommand>(ba, Action::deposit, 100);
  deposit100->call(); // 500
  deposit100->call(); // 600
  deposit100->undo(); // 500
  std::cout << ba << std::endl;
  return 0;
}
