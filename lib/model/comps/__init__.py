
import threading
from collections import defaultdict

CONTEXT = threading.local()
CONTEXT.recorders = []
CONTEXT.BPTT_limit = 10

# ----------------------------------------------------------------------------------------------------------------------#
# HISTORY BASE CLASS


# Moments are responsible for holding variables needed for propagating back through time.
class Moment:
    def __init__(self):
        self.is_record = False # Moments can be records or "on the fly creations" provided by the defaultdict...
    pass


class Recorder:
    def __init__(self, default_lambda):
        self.default_lambda = default_lambda
        self.history: dict = None  # history
        self.reset()
        CONTEXT.recorders.append(self)
        self.time_restrictions : list = None

    def reset(self): self.history: dict = defaultdict(self.default_lambda)  # history

    def latest(self, time: int):
        if self.time_restrictions is not None: assert time in self.time_restrictions
        moment = self.history[time]
        is_record = moment.is_record
        while not is_record:
            time -= 1
            moment = self.history[time]
            is_record = moment.is_record
            if time < 0: # first time step is recorded by nodes!
                self.history[-1] = self.history[-1]
                self.history[-1].is_record = True
                return self.history[-1]

        return self.history[time]

    def at(self, time:int):
        if self.time_restrictions is not None: assert time in self.time_restrictions
        return self.history[time]

    def rec(self, time: int):
        if self.time_restrictions is not None: assert time in self.time_restrictions
        if not self.history[time].is_record:
            self.history[time] = self.history[time]
            self.history[time].time = time
        self.history[time].is_record = True
        return self.history[time]

