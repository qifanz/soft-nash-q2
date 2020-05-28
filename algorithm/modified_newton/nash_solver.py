import math
import pickle
import warnings

from util.matrix_game_solver import *

warnings.filterwarnings('ignore', '.*Ill-conditioned*')

DEBUG = False


class NashSolver:
    def __init__(self, env, nash_value_file, nash_policy_file, precision=4):
        self.env = env
        self.value_file = nash_value_file
        self.policy_file = nash_policy_file

        self.alpha = 0.5  # convergence factor
        self.mu = 0.6
        self.gamma = 0.9
        self.beta = 0.9
        self.value_vector = env.rewards  # set initial estimation to rewards
        self.precision = precision

    def solve(self):
        converge_flag = False
        iteration = 0
        while not converge_flag:
            iteration += 1
            L_v, policy = self.__calc_L()
            psi_v = self.__calc_psi(L_v, self.value_vector)
            J_v = self.__calc_J(psi_v)
            if J_v <= math.pow(10, -self.precision):
                converge_flag = True
            else:
                D_k, I_subtract_P = self.__cal_D(policy, psi_v)
                w = 1
                k = 0
                while True:
                    if self.__test_inequality(D_k, w, J_v, psi_v, I_subtract_P):
                        self.value_vector = np.add(self.value_vector, w * D_k)
                        break
                    else:
                        print('Test inequality failed, k = ', k)
                        w = self.mu * w
                        k += 1
            print('iteration ', iteration)
        for i, v in enumerate(self.value_vector):
            if v < math.pow(10, -self.precision):
                self.value_vector[i] = 0
        f = open(self.value_file, "wb")
        pickle.dump(self.value_vector, f)
        f.close()
        f = open(self.policy_file, "wb")
        pickle.dump(policy, f)
        f.close()
        return self.value_vector, policy

    def create_action_value_matrix(self, state, L_v, use_Lv):
        if use_Lv:
            value_vector = L_v
        else:
            value_vector = self.value_vector
        immediate_reward = self.env.get_state_reward(state)
        action_value_matrix = np.ones((self.env.get_n_actions(), self.env.get_n_actions())) * immediate_reward

        for i in range(self.env.get_n_actions()):
            for j in range(self.env.get_n_actions()):
                transition_vector = self.env.get_state_transition((i, j))[state]
                action_value_matrix[i, j] += self.gamma * np.dot(transition_vector, value_vector)
        return action_value_matrix

    def __solve_state(self, state, L_v=None, use_Lv=False):
        if self.env.is_terminal_state(state):
            return np.zeros(self.env.get_n_actions()), np.zeros(self.env.get_n_actions()), self.env.get_state_reward(
                state)
        start = time.time()
        action_value_matrix = self.create_action_value_matrix(state, L_v, use_Lv)
        end = time.time()
        self.debug(['Create M used', end - start])
        start = time.time()
        value, policy_x, policy_y = linprog_solve(np.array(action_value_matrix), precision=self.precision)
        end = time.time()
        self.debug(['Linprog solver used', end - start])
        return policy_x, policy_y, value

    def __calc_L(self, new_v=None, use_new_v=False):
        L_v = []
        policy = []
        for state in range(self.env.get_n_states()):
            policy_x, policy_y, value = self.__solve_state(state, new_v, use_new_v)
            L_v.append(value)
            policy.append((policy_x, policy_y))
        return L_v, policy

    def __calc_psi(self, L_v, v):
        return np.subtract(L_v, v)

    def __calc_J(self, psi_v):
        return 0.5 * np.dot(psi_v.T, psi_v)

    def __cal_D(self, policy, psi_v):
        n_states = self.env.get_n_states()
        I = np.identity(n_states)
        for state in range(self.env.get_n_states()):
            policy_x = policy[state][0]
            policy_y = policy[state][1]
            for x in range(self.env.get_n_actions()):
                for y in range(self.env.get_n_actions()):
                    I[state] = np.subtract(I[state], self.beta * policy_x[x] * policy_y[y] *
                                           self.env.get_state_transition((x, y))[state])

        return np.dot(np.linalg.inv(I), psi_v), I

    def __calc_delta_J(self, psi_v, I_subtract_P):
        return -np.dot(psi_v.T, I_subtract_P)

    def __test_inequality(self, d_k, w, j_v, psi_v, I_subtract_P):
        new_v = self.value_vector + d_k * w
        new_l, new_policy = self.__calc_L(new_v, True)
        new_psi = self.__calc_psi(new_l, new_v)
        left = self.__calc_J(new_psi) - j_v
        right = self.alpha * w * np.dot(self.__calc_delta_J(psi_v, I_subtract_P), d_k)
        return left <= right

    def debug(self, strs):
        if DEBUG:
            line = ''
            for s in strs:
                line += str(s)
                line += ' '
            print(line)
