import pickle

from util.matrix_game_solver import *


class Q:
    def __init__(self, n_states, n_actions, discount_factor, lr, use_prior=False):
        if use_prior:
            f = open('../data/nash_q_prior_values.pkl', 'rb')
            self.Q = pickle.load(f)
            f.close()
        else:
            self.Q = np.zeros((n_states, n_actions, n_actions))
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

    def update(self, reward, state, action, action_op, new_state, precision=4):
        # only update once, when its "Player"
        if new_state is None:
            self.Q[state] = np.ones((self.n_actions, self.n_actions)) * reward
            return np.divide(np.ones(self.n_actions), self.n_actions), np.divide(np.ones(self.n_actions), self.n_actions)
        else:
            value_new_state, px, py = linprog_solve(self.get_matrix_game(new_state), precision)
            self.Q[state, action, action_op] += self.lr * (
                    reward + self.discount_factor * value_new_state - self.Q[
                state, action, action_op])
            px = np.divide(px, np.sum(px))
            px = np.nan_to_num(px)
            py = np.divide(py, np.sum(py))
            py = np.nan_to_num(py)
            return px, py
