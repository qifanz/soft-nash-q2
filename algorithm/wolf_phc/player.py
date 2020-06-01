import numpy as np


class Player:
    def __init__(self, index, n_states, n_actions, delta_w, delta_l, lr, epsilon=0.3, update_frequency=100):
        self.index = index
        self.n_states = n_states
        self.n_actions = n_actions
        self.delta_w = delta_w
        self.delta_l = delta_l
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))
        self.policies = []
        self.average_policy = []
        for state in range(self.n_states):
            self.policies.append(self.normalize(np.ones(self.n_actions)))
            self.average_policy.append(self.normalize(np.ones(self.n_actions)))
        self.counter = np.zeros(n_states)
        self.lr = lr
        self.update_frequency = update_frequency

    def normalize(self, policy):
        policy = np.divide(policy, np.sum(policy))
        policy = np.nan_to_num(policy)
        return policy

    def get_policy(self, state):
        return self.policies[state]

    def get_average_policy(self, state):
        return self.average_policy[state]

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            policy = self.normalize(np.ones(self.n_actions))
        else:
            policy = self.get_policy(state)
        return np.random.choice(np.arange(0, self.n_actions), p=policy)

    def observe(self, state, action, reward, next_state):
        self.update_q(state, action, reward, next_state)
        self.update_average_policy(state)
        self.update_policy(state, action)

    def update_q(self, state, action, reward, next_state):
        self.counter[state] += 1
        if state is None:
            self.Q[state] = np.ones((self.n_actions, self.n_actions)) * reward
        else:
            lr_decreased = 1 / (1 + int(+self.counter[state] / self.update_frequency)) * self.lr
            # otherwise it decays too fast!
            self.Q[state, action] = (1 - lr_decreased) * self.Q[state, action] + lr_decreased * (
                    reward + 0.9 * np.max(self.Q[next_state]))

    def update_average_policy(self, state):
        count_state = self.counter[state]
        average_policy_state = self.get_average_policy(state)
        for action in range(self.n_actions):
            average_policy_state[action] = average_policy_state[action] * (
                    count_state - 1) / count_state + 1 / count_state * self.get_policy(state)[action]
        self.average_policy[state] = self.normalize(average_policy_state)

    def update_policy(self, state, action):
        should_increment = action == np.argmax(self.Q[state])
        use_delta_w = np.dot(self.get_policy(state), self.Q[state]) > np.dot(self.get_average_policy(state),
                                                                             self.Q[state])
        if use_delta_w:
            delta = self.delta_w
        else:
            delta = self.delta_l
        delta = 1 / (1 + int(+self.counter[state] / self.update_frequency)) * delta  # same, otherwise it decays too fast!
        if should_increment:
            self.policies[state][action] += delta
        else:
            self.policies[state][action] -= delta / (self.n_actions - 1)
            self.policies[state][action] = max(0, self.policies[state][action])
        self.policies[state] = self.normalize(self.policies[state])

    def get_all_policies(self):
        policies = []
        for state in range(self.n_states):
            policies.append(self.get_policy(state))
        return policies
