import numpy as np

class SingleQPlayer:
    def __init__(self, Q, n_actions, epsilon=0):
        # For now assume its a 2-player env
        self.Q = Q
        self.n_actions = n_actions
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.uniform() >= self.epsilon:
            return self.Q.get_state_best_action(state)
        else:
            return np.random.choice(np.arange(0, self.n_actions), p=np.divide(np.ones(self.n_actions), self.n_actions))

    def get_policy(self, state):
        policy = np.zeros(self.n_actions)
        action = self.Q.get_state_best_action(state)
        policy[action] = 1
        return policy

    def observe(self, reward, state, action_self, action_op, new_state):
        self.Q.update(reward, state, action_self, new_state)

    def generate_all_policies(self, n_states):
        policies = []
        for state in range(n_states):
            policies.append(self.get_policy(state))
        return policies

