// https://tamir.dev/posts/a-functional-style-state-machine-in-cpp/

#include <cstdio>
#include <cstdlib>
enum class Event { A, B };

struct Context {
  int counter = 0;
};

struct State {
  using FuncType = State (*)(Context&, Event);

  State(FuncType f) : func_(f) {}
  State operator()(Context& ctx, Event evt) { return func_(ctx, evt); }

 private:
  FuncType func_;
};

// declare functions that accept Ctx and Event and return state

State A(Context&, Event);
State B(Context&, Event);

State A(Context& ctx, Event evt) {
  printf("State A, counter = %d\n", ctx.counter);
  ++ctx.counter;
  switch (evt) {
    case Event::A:
      return A;
    case Event::B:
      return B;
    default:
      abort();
  }
}

State B(Context& ctx, Event evt) {
  printf("State B, counter = %d\n", ctx.counter);
  ++ctx.counter;
  switch (evt) {
    case Event::A:
      return A;
    case Event::B:
      return B;
    default:
      abort();
  }
}

int main(int argc, char const* argv[]) {
  State state = A;
  Context ctx{};
  Event events[] = {Event::B, Event::A, Event::B, Event::A};

  for (auto evt : events) {
    state = state(ctx, evt);
  }

  return 0;
}