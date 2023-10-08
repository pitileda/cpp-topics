#pragma once

#include <iostream>

#ifdef UNIT_TESTS
class BarTest;
#endif

class Bar {
  // #ifndef UNIT_TESTS
  //  private:
  //   Bar() = default;
  //   ~Bar(){};
  // #endif

 public:
  // #ifdef UNIT_TESTS
  //   friend void BarTest_HandleDoTest_Positive2();
  // #endif
  virtual int Do(int a);

  // #ifndef UNIT_TESTS
  //  private:
  // #endif
  virtual int Oops(int b);
};