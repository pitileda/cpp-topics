class IncrementerBase {
 public:
  IncrementerBase() {
    my_num = rand();
  }
  virtual ~IncrementerBase() {}
  virtual void Increment() {}
  int my_num;
};

class IncrementerVirtual : public IncrementerBase {
 public:
  IncrementerVirtual() { }
  virtual ~IncrementerVirtual() {}

  virtual void Increment() override {
    g_update_count += my_num;
  }
};

class IncrementerDirect : public IncrementerBase {
 public:
  IncrementerDirect() {}
  virtual ~IncrementerDirect() {}

  // Preventing inling to force a fair comparison.
  __declspec(noinline) void IncrementDirect() {
    g_update_count += my_num;
  }
};