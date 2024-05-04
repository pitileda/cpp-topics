from abc import ABC, abstractmethod


class State(object):
    """docstring for State"""

    def __init__(self, arg):
        super(State, self).__init__()
        self.arg = arg

    @abstractmethod
    def on_event(self) -> None:
        pass


class Idle(State):
    """Idle state"""

    def __init__(self, arg):
        super(Idle, self).__init__()
        self.arg = arg

    def on_event(self) -> None:
        pass
