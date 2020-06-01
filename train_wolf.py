from algorithm.wolf_phc.train import *
from environment.PEG_game import *
from environment.low_dim_PEG import *
from util.policy_evaluator import PolicyEvaluator

env = LowDimPEG()
game = PEGGame(env)
ground_truth_policies_file = 'data/LowDimensionPEG/nash_values.pkl'
ground_truth_values_file = 'data/LowDimensionPEG/nash_policies.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
total_run = 5
for i in range(total_run):
    train(game,
          evaluator,
          200000, 2000, delta_l=0.1,delta_w=0.4)
