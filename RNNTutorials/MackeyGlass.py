import math
import numpy as np


class MackeyGlass:
    def __init__(self, n=10, beta=0.2, gamma=0.1, tau=30):
        self.n = n
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

        self.time_history = np.random.random((1, self.tau))

    def get_next_value(self):
        x_now = self.time_history[0]
        x_end = self.time_history[-1]
        x_next = x_now + (self.beta * (x_end / (1.0 + pow(x_end, self.n)))) - (self.gamma * x_now)

        # Put the new value to the front of the list and then due to the size increase, delete
        # the last value so we remain at size 'tau'
        self.time_history = np.insert(self.time_history, 0, x_next, axis=0)
        self.time_history = np.delete(self.time_history, (self.tau-1))

        return x_next

    def generate_samples(self, num_samples):
        x = []
        [x.append(self.get_next_value()) for i in range(0, num_samples)]

        return x

