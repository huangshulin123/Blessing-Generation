import torch.optim


class ScheduledOptim(object):
    "Optim wrapper that implements rate."
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps, max_lr=5e-4, min_lr=3e-5, beta=0.55):
        self.__optimizer = optimizer

        self._step = 0
        self._rate = 0

        self.__warmup_steps = warmup_steps
        self.__max_lr = max_lr
        self.__min_lr = min_lr

        self.__alpha = warmup_steps**(-beta-1.0)
        self.__beta = -beta

        self.__scale = 1.0 / (self.__alpha*warmup_steps)


    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.__optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.__optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        lr = self.__max_lr*self.__scale*min(step*self.__alpha, step**(self.__beta))
        if step > self.__warmup_steps:
            lr = max(lr, self.__min_lr)
        return lr

    def zero_grad(self):
        self.__optimizer.zero_grad()

    def state_dict(self):
        return self.__optimizer.state_dict()

    def load_state_dict(self, dic):
        self.__optimizer.load_state_dict(dic)
