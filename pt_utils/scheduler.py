class CustomSchedule:
    """
    A simple wrapper class for learning rate scheduling
    from git@github.com:jadore801120/attention-is-all-you-need-pytorch.git
    https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py
    """

    def __init__(self, optimizer, decay_step, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.decay_step = decay_step
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
        # for param_group in self._optimizer.param_groups:
        #     self.lr = param_group['lr']

    def step_and_update_lr(self):
        """"
        Step with the inner optimizer
        """
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        """
        Zero out the gradients with the inner optimizer
        """
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * (n_warmup_steps ** (-1.5)))

    def _update_learning_rate(self):
        """
        Learning rate scheduling per step
        """

        self.n_steps += 1
        if self.n_steps > self.decay_step:
            lr = 1e-4 #* self.lr
        else:
            lr = self._get_lr_scale() #* self.lr
        # print("step:",self.n_steps, "\nlr:",lr,"\n")
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            # break