import threading

# Timer that can be started and reset
class ResettableTimer:
    def __init__(self, delay: float, function : callable) -> None:
        self.delay = delay
        self.function = function
        self.timer = None

    def start(self) -> None:
        self.cancel()
        self.timer = threading.Timer(self.delay, self.function)
        self.timer.start()

    def cancel(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None