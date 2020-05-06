import torch


class GECO():

    def __init__(self, goal, step_size, alpha=0.99, beta_init=1.0,
                 beta_min=1e-10, speedup=None):
        self.err_ema = None
        self.goal = goal
        self.step_size = step_size
        self.alpha = alpha
        self.beta = torch.tensor(beta_init)
        self.beta_min = torch.tensor(beta_min)
        self.beta_max = torch.tensor(1e10)
        self.speedup = speedup

    def to_cuda(self):
        self.beta = self.beta.cuda()
        if self.err_ema is not None:
            self.err_ema = self.err_ema.cuda()

    def loss(self, err, kld):
        # Compute loss with current beta
        loss = err + self.beta * kld
        # Update beta without computing / backpropping gradients
        with torch.no_grad():
            if self.err_ema is None:
                self.err_ema = err
            else:
                self.err_ema = (1.0-self.alpha)*err + self.alpha*self.err_ema
            constraint = (self.goal - self.err_ema)
            if self.speedup is not None and constraint.item() > 0:
                factor = torch.exp(self.speedup * self.step_size * constraint)
            else:
                factor = torch.exp(self.step_size * constraint)
            self.beta = (factor * self.beta).clamp(self.beta_min, self.beta_max)
        # Return loss
        return loss
