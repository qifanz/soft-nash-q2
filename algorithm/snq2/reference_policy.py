import pickle

from util.matrix_game_solver import *

THRESHOLD = 0.1


class ReferencePolicy:
    def __init__(self, n_states, n_actions, strategy='uniform', prior_file=''):
        self.n_states = n_states
        self.n_actions = n_actions
        self.reference_policy = [[], []]
        self.strategy = strategy
        self.prior_file = prior_file
        self.init_reference()

    def init_reference(self):
        if self.strategy == 'quasi-nash':
            f = open(self.prior_file, 'rb')
            policies = pickle.load(f)
            f.close()
            for policy in policies:
                self.reference_policy[0].append(policy[0])
                self.reference_policy[1].append(policy[1])
        for state in range(self.n_states):
            if self.strategy == 'uniform':
                self.reference_policy[0].append(np.multiply(np.ones(self.n_actions), 1 / self.n_actions))
                self.reference_policy[1].append(np.multiply(np.ones(self.n_actions), 1 / self.n_actions))
            elif self.strategy == 'random':
                ref0 = []
                ref1 = []
                for action in range(self.n_actions):
                    ref0.append(np.random.uniform())
                    ref1.append(np.random.uniform())
                self.reference_policy[0].append(np.divide(ref0, np.sum(ref0)))
                self.reference_policy[1].append(np.divide(ref1, np.sum(ref1)))
            elif self.strategy == 'quasi-uniform':
                ref0 = []
                ref1 = []
                for action in range(self.n_actions):
                    ref0.append(np.random.uniform(0, 0.1))
                    ref1.append(np.random.uniform(0, 0.1))
                tmp0 = np.add(ref0, np.multiply(np.ones(self.n_actions), 1 / self.n_actions))
                tmp1 = np.add(ref1, np.multiply(np.ones(self.n_actions), 1 / self.n_actions))
                self.reference_policy[0].append(np.divide(tmp0, np.sum(tmp0)))
                self.reference_policy[1].append(np.divide(tmp1, np.sum(tmp1)))
            elif self.strategy == 'quasi-nash':

                ref0 = []
                ref1 = []
                for action in range(self.n_actions):
                    ref0.append(np.random.uniform(0, 0.1))
                    ref1.append(np.random.uniform(0, 0.1))
                self.reference_policy[0][state] = np.add(self.reference_policy[0][state], ref0)
                self.reference_policy[1][state] = np.add(self.reference_policy[1][state], ref1)
                self.reference_policy[0][state] = np.divide(self.reference_policy[0][state],
                                                            np.sum(self.reference_policy[0][state]))
                self.reference_policy[1][state] = np.divide(self.reference_policy[1][state],
                                                            np.sum(self.reference_policy[1][state]))
                '''
                '''
                for action in range(self.n_actions):
                    if self.reference_policy[0][state][action] < 0.01:
                        self.reference_policy[0][state][action] = 0.01
                    if self.reference_policy[1][state][action] < 0.01:
                        self.reference_policy[1][state][action] = 0.01

                self.reference_policy[0][state] = self.normalize(self.reference_policy[0][state])
                self.reference_policy[1][state] = self.normalize(self.reference_policy[1][state])
            else:
                raise Exception('Cannot recognize reference strategy')

    def normalize(self, policy):
        return np.divide(policy, np.sum(policy))

    def get_reference_state(self, index, state):
        return self.reference_policy[index][state]

    def get_reference_state_action(self, index, state, action):
        return self.reference_policy[index][state][action]

    def update_reference(self, Q, factor, is_terminal_state):
        superior_threshold_count = 0
        inferior_threshold_count = 0
        for state in range(self.n_states):
            if is_terminal_state(state):
                continue
            matrix_game = Q.get_matrix_game(state)
            old_value = np.dot(np.dot(self.reference_policy[0][state], matrix_game), self.reference_policy[1][state])
            value, policy_1, policy_2 = linprog_solve(matrix_game)
            self.reference_policy[0][state] = np.divide(
                np.add(np.multiply(self.reference_policy[0][state], factor), policy_1), factor + 1)
            self.reference_policy[1][state] = np.divide(
                np.add(np.multiply(self.reference_policy[1][state], factor), policy_2), factor + 1)
            deviation = np.fabs(np.fabs(old_value - value) / value)
            if deviation <= THRESHOLD:
                inferior_threshold_count += 1
            else:
                superior_threshold_count += 1
        return inferior_threshold_count > 4 * superior_threshold_count  # if 80% are < threshold, return True: two updates are close enough

    def get_reference(self, index):
        return self.reference_policy[index]
