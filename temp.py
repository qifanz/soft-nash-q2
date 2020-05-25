import pickle
import numpy as np
from environment.low_dim_PEG import LowDimPEG
from util.policy_evaluator import PolicyEvaluator
from util.matrix_game_solver import *
import math
env = LowDimPEG()
ground_truth_policies_file = 'data/LowDimensionPEG/nash_values.pkl'
ground_truth_values_file = 'data/LowDimensionPEG/nash_policies.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)

policy_file = 'data/LowDimensionPEG/nashq_policies.pkl'
f = open(policy_file, 'rb')
policy = pickle.load(f)
f.close()
for s in range(256):
    if env.is_terminal_state(s):
        continue
    matrix_game = evaluator.create_matrix_game(s)
    correct_policy = [evaluator.nash_policies[0][s],evaluator.nash_policies[1][s]]
    current_policy = [policy[0][s],policy[1][s]]
    nash_value = value_solve(matrix_game,10)
    value = np.dot(np.dot(policy[0][s], matrix_game), policy[1][s].T)
    value2 = max(np.dot(matrix_game, policy[1][s]))
    value3 = min(np.dot(policy[0][s], matrix_game))

    diff = math.fabs(math.fabs(value - nash_value) / nash_value)
    diff1 = math.fabs(math.fabs(nash_value - value2) / nash_value)
    diff2 = math.fabs(math.fabs(nash_value - value3) / nash_value)

    if not (diff<0.01 and diff1<0.01 and diff2<0.01):
        print('state',str(s))
        print(diff)
        print(diff1)
        print(diff2)
        print(matrix_game)
        print(policy[0][s], policy[1][s])
        print(evaluator.nash_policies[0][s],evaluator.nash_policies[1][s])
