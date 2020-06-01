import numpy as np


class Q_table:
    def __init__(self, n_states, n_actions, discount_factor, lr):
        self.Q = np.zeros((n_states, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.lr = lr

    def get_value(self, state, action):
        return self.Q[state, action]

    def get_state_best_action(self, state):
        return np.argmax(self.Q[state])

    def update(self, reward, state, action, new_state):
        self.Q[state, action] = (1 - self.lr) * self.Q[state, action] + self.lr * (
                    reward + self.discount_factor * np.max(self.Q[new_state]))
