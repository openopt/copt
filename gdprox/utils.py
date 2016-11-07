from datetime import datetime

class Trace:
    def __init__(self, loss_func):
        self.loss_func = loss_func
        self.vals = []
        self.times = []
        self.start = datetime.now()

    def __call__(self, args):
        fxk = self.loss_func(args)
        print(fxk)
        self.vals.append(fxk)
        self.times.append((datetime.now() - self.start).total_seconds())