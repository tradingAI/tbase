import numpy as np


class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1,
                 sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(
            mu=mu, sigma=sigma, sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.current_sigma * np.sqrt(self.dt) * np.random.normal(
                size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


class GaussianProcess:
    def __init__(self, mu=0., sigma=0.5, size=1, decay=0.9, decay_steps=10000):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.decay = decay
        self.decay_steps = decay_steps
        self.n_steps = 0

    def sample(self):
        self.n_steps += 1
        if self.n_steps % self.decay_steps == 0:
            self.sigma = self.sigma * self.decay
        x = np.random.normal(loc=self.mu, scale=self.sigma, size=self.size)
        return x


if __name__ == '__main__':
    ou = OrnsteinUhlenbeckProcess(0.1, size=4)
    states = []
    for i in range(1000):
        states.append(ou.sample())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()
