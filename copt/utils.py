from datetime import datetime


class Trace:
    """
    XXX
    """
    def __init__(self, loss_func, print_freq=100, verbose=False):
        self.loss_func = loss_func
        self.vals = []
        self.times = []
        self.start = None
        self.counter = 0
        self.print_freq = print_freq
        self.verbose = verbose

    def __call__(self, args):
        fxk = self.loss_func(args)
        if self.verbose and self.counter % self.print_freq == 0:
            print('Iteration: %s, Trace obj: %s' %
                  (self.counter, fxk))
        self.counter += 1
        self.vals.append(fxk)
        if self.start is None:
            self.start = datetime.now()
            self.times = [0]
        else:
            self.times.append((datetime.now() - self.start).total_seconds())
