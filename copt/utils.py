from datetime import datetime


class Trace:
    """
    XXX
    """
    def __init__(self, loss_func, print_freq=100):
        self.loss_func = loss_func
        self.vals = []
        self.times = []
        self.start = datetime.now()
        self.counter = 0
        self.print_freq = 100

    def __call__(self, args):
        fxk = self.loss_func(args)
        if self.counter % self.print_freq == 0:
            print('Iteration: %s, Trace obj: %s' %
                  (self.counter, fxk))
        self.counter += 1
        self.vals.append(fxk)
        self.times.append((datetime.now() - self.start).total_seconds())