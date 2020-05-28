from util.matrix_game_solver import *


class DoubleQ:
    def __init__(self, n_states, n_actions, discount_factor, lr):
        self.Q = np.zeros((n_states, n_actions, n_actions))
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount_factor = discount_factor
        self.lr = lr

    def set_players(self, player, op):
        self.player = player
        self.op = op

    def get_value(self, state, action, action_op, player_index):
        if player_index == 0:
            return self.Q[state, action, action_op]
        else:
            return self.Q[state, action_op, action]

    def get_matrix_game(self, state):
        return self.Q[state]

    def update(self, reward, state, action, action_op, new_state, player, using_nash=False, precision=4):
        if player.index == 0:
            # only update once, when its "Player"
            if new_state is not None:
                if not using_nash:
                    self.Q[state, action, action_op] += self.lr * (
                            reward + self.discount_factor * self.estimate_value_of_state(new_state) - self.Q[
                        state, action, action_op])
                else:
                    self.Q[state, action, action_op] += self.lr * (
                            reward + self.discount_factor * self.estimate_nash_of_state(new_state, precision=precision) - self.Q[
                        state, action, action_op])
            else:
                self.Q[state] = np.ones((self.n_actions, self.n_actions)) * reward

    def estimate_value_of_state(self, state):
        matrix_game = self.get_matrix_game(state)
        policy_player = self.player.get_policy(state)
        policy_op = self.op.get_policy(state)
        value = np.dot(np.dot(policy_player, matrix_game), policy_op)
        return value

    def estimate_nash_of_state(self, state, precision=4):
        matrix_game = self.get_matrix_game(state)
        value = value_solve(matrix_game,precision=precision)
        return value
