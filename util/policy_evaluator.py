import math
import pickle

import numpy as np

SOFTQ_POLICY_FILE = '../data/policies_single_q.pkl'


class PolicyEvaluator:
    def __init__(self, env, ground_truth_policies_file, ground_truth_values_file, gamma=0.9, threshold=0.03):
        self.env = env
        self.is_srps = False
        if not env.get_name() == 'sRPS':
            self.nash_values = self.load_nash_values(ground_truth_policies_file)
            self.nash_policies = self.load_nash_policy(ground_truth_values_file)
        else:
            self.is_srps = True
        self.gamma = gamma
        self.threshold = threshold

    def validate(self, polciy2Evaluate, debug=False):
        correct_states = []
        incorrect_states = []
        deviations = []
        for state in range(self.env.get_n_states()):
            if self.env.is_non_terminal_state(state):
                is_correct, deviation = self.validate_state(state, polciy2Evaluate)
                if is_correct:
                    correct_states.append(state)
                else:
                    incorrect_states.append(state)
                deviations.append(deviation)
                if debug:
                    if deviation > 1:
                        print(state)
                        print(polciy2Evaluate[0][state], polciy2Evaluate[1][state])
                        print(self.nash_policies[0][state], self.nash_policies[1][state])
                        print(self.create_matrix_game(state))

        return correct_states, incorrect_states, deviations

    def validate_state(self, state, polciy2Evaluate):
        if self.is_srps:
            correct_pl = max(polciy2Evaluate[0][state]) < 0.338 and min(polciy2Evaluate[0][state]) > 0.328
            correct_op = max(polciy2Evaluate[1][state]) < 0.338 and min(polciy2Evaluate[1][state]) > 0.328
            deviation = max(max(np.fabs(0.33333 - polciy2Evaluate[0][state]) / 0.33333),
                            max(np.fabs(0.33333 - polciy2Evaluate[1][state]) / 0.33333))
            return correct_op and correct_pl, deviation
        matrix_game = self.create_matrix_game(state)
        nash_value = self.nash_values[state]
        value = np.dot(np.dot(polciy2Evaluate[0][state], matrix_game), polciy2Evaluate[1][state].T)
        value2 = max(np.dot(matrix_game, polciy2Evaluate[1][state]))
        value3 = min(np.dot(polciy2Evaluate[0][state], matrix_game))
        if math.fabs(nash_value) <= 0.01:
            diff = math.fabs(value)
            diff1 = math.fabs(value2)
            diff2 = math.fabs(value3)
        elif math.fabs(value) <= 0.01:
            diff = math.fabs(nash_value)
            diff1 = math.fabs(nash_value)
            diff2 = math.fabs(nash_value)
        else:
            diff = math.fabs(math.fabs(value - nash_value) / nash_value)
            diff1 = math.fabs(math.fabs(nash_value - value2) / nash_value)
            diff2 = math.fabs(math.fabs(nash_value - value3) / nash_value)

        return diff < self.threshold and diff1 < self.threshold and diff2 < self.threshold, max(diff, diff1, diff2)

    def create_matrix_game(self, state):
        n_actions = self.env.get_n_actions()
        action_value_matrix = np.ones((n_actions, n_actions)) * self.env.rewards[state]

        for i in range(n_actions):
            for j in range(n_actions):
                transition_vector = self.env.get_state_transition((i, j))[state]
                action_value_matrix[i, j] += self.gamma * np.dot(
                    transition_vector, self.nash_values)
        return action_value_matrix

    def load_nash_policy(self, file):
        f = open(file, 'rb')
        policies = pickle.load(f)
        f.close()
        p_x = []
        p_y = []
        for policy in policies:
            p_x.append(policy[0])
            p_y.append(policy[1])

        return p_x, p_y

    def load_softq_policy(self, file=SOFTQ_POLICY_FILE):
        f = open(file, 'rb')
        policies = pickle.load(f)
        f.close()

        return policies[0], policies[1]

    def load_nash_values(self, file):
        f = open(file, "rb")
        values = pickle.load(f)
        f.close()
        return values
