from algorithm.wolf_phc.train import *
from environment.srps import RPSEnv
from util.policy_evaluator import PolicyEvaluator

game = RPSEnv()
init_update_freq = 25
ground_truth_policies_file = None
ground_truth_values_file = None
evaluator = PolicyEvaluator(game, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.5
total_run = 5
for i in range(total_run):
    info = '| init_lr:' + str(init_lr) + '| No.' + str(i + 1) + '/' + str(total_run)
    train(game,evaluator,100000,1000, update_frequency=10, delta_l=0.04, delta_w=0.01)


