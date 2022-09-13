class kl_annealing:
        def __init__(self, args):
            # super().__init__()
            self.epochs = args.niter
            self.epoch_size = args.epoch_size
            self.mode = args.kl_anneal_cyclical
            self.ratio = args.kl_anneal_ratio
            self.cycles = args.kl_anneal_cycle if self.mode else 1
            self.period = self.epochs / self.cycles
            self.step = 1.0 / (self.period * self.ratio) # within [0, 1]
            self.v = 0.0
            self.i = 0
            self.t = 0
        
        def update(self):
            self.v += self.step
            self.i += 1
            if self.i == self.period:
                self.v = 0.0
                self.i = 0
            if self.v > 1.0:
                self.v = 1.0
        
        def get_beta(self):
            beta = 1.0 / (1.0 + np.exp(-(self.v * 12.0 - 6.0))) if self.mode else self.v
            self.t += 1
            if self.t == self.epoch_size:
                self.t = 0
                self.update()
            return beta