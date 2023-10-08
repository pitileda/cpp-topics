#pragma once

#include <iostream>

class Bar {
 public:
  // #ifdef UNIT_TESTS
  //   friend void BarTest_HandleDoTest_Positive2();
  // #endif
  virtual int Do(int a);

 private:
  virtual int Oops(int b);
};