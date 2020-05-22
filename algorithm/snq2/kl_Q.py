import math

import numpy as np


class QWithKL:
    def __init__(self, n_states, n_actions, discount_factor, lr):
        self.Q = np.ones((n_states, n_actions, n_actions)) * 2
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.lr = lr

    def get_value(self, state, action, action_op, player_index):
        if player_index == 0:
            return self.Q[state, action, action_op]
        else:
            return self.Q[state, action_op, action]

    def get_matrix_game(self, state):
        return self.Q[state]

    def update(self, reward, state, action, action_op, new_state, player):
        if new_state is None:
            # Terminal state
            self.Q[state] = np.ones((self.n_actions, self.n_actions)) * reward
        else:
            self.Q[state, action, action_op] += self.lr * (
                    reward + self.discount_factor * self.calc_value_of_state(new_state, player) - self.Q[
                state, action, action_op])

    def calc_value_of_state(self, state, player):
        sum = 0
        for action in range(self.n_actions):
            sum += player.get_reference_self(state, action) * math.exp(player.beta * player.marginalize(state, action))
        sum = math.log(sum) / player.beta
        return sum
