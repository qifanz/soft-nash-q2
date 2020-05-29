import math

from algorithm.snq2.reference_policy import *


class Player:
    def __init__(self, index, kl_q, q,
                 n_actions,
                 beta,
                 beta_op,
                 reference_policy: ReferencePolicy,
                 epsilon=0,
                 use_double_q=True):

        self.kl_q = kl_q
        self.q = q
        self.index = index
        self.n_actions = n_actions
        self.beta = beta
        self.beta_op = beta_op
        self.epsilon = epsilon
        self.reference_policy = reference_policy
        self.use_double_q = use_double_q

    def choose_action(self, state):
        if np.random.uniform() >= self.epsilon:
            return np.random.choice(np.arange(0, self.n_actions), p=self.get_policy(state))
        else:
            return np.random.choice(np.arange(0, self.n_actions), p=np.divide(np.ones(self.n_actions), self.n_actions))

    def marginalize(self, state, action):
        sum = 0
        for a_op in range(self.n_actions):
            sum += self.get_reference_op(state, a_op) * math.exp(
                self.beta_op * self.kl_q.get_value(state, action, a_op, self.index))
        return 1 / self.beta_op * math.log(sum)

    def get_policy(self, state):
        action_possibility = []
        normalizer = 0
        for action in range(self.n_actions):
            prob = self.get_reference_self(state, action) * math.exp(self.beta * self.marginalize(state, action))
            action_possibility.append(prob)
        policy = np.divide(action_possibility, np.sum(action_possibility))
        for i,p in enumerate(policy):
            if p<10e-8:
                policy[i]=0
        policy = np.divide(policy, np.sum(policy))
        return np.nan_to_num(policy)

    def get_reference_self(self, state, action):
        return self.reference_policy.get_reference_state_action(self.index, state, action)

    def get_reference_op(self, state, action):
        return self.reference_policy.get_reference_state_action(not self.index, state, action)

    def generate_all_policies(self, n_states):
        policies = []
        for state in range(n_states):
            policies.append(self.get_policy(state))
        return policies
