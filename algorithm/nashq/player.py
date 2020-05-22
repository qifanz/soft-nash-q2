from util.matrix_game_solver import *


class NashPlayer:
    def __init__(self, index, Q, n_actions, epsilon=0):
        # For now assume its a 2-player env
        self.Q = Q
        self.index = index
        self.n_actions = n_actions
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform() >= self.epsilon:
            return np.random.choice(np.arange(0, self.n_actions), p=self.get_policy(state))
        else:
            return np.random.choice(np.arange(0, self.n_actions), p=np.divide(np.ones(self.n_actions), self.n_actions))

    def get_policy(self, state):
        M = self.Q.get_matrix_game(state)

        # res[0] = value, res[1] = player, res[2] = opponent
        res = linprog_solve(M)
        policy = res[self.index + 1]
        policy = np.divide(policy, np.sum(policy))
        policy = np.nan_to_num(policy)
        return policy

    def observe(self, reward, state, action_self, action_opponent, new_state):
        self.Q.update(reward, state, action_self, action_opponent, new_state, self)

    def generate_all_policies(self, n_states):
        policies = []
        for state in range(n_states):
            policies.append(self.get_policy(state))
        return policies
