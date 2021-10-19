from algorithm.wolf_phc.train import train
from environment.PEG_game import PEGGame
from environment.det_peg import DetPEG
from environment.high_dim_PEG import HighDimEnv
from environment.low_dim_PEG import LowDimPEG
from util.policy_evaluator import PolicyEvaluator

env = LowDimPEG()
game = PEGGame(env)
ground_truth_policies_file = 'data/'+env.get_name()+'/nash_values.pkl'
ground_truth_values_file = 'data/'+env.get_name()+'/nash_policies.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.5
total_run = 5
for i in range(total_run):
    info = '| init_lr:' + str(init_lr) + '| No.' + str(i + 1) + '/' + str(total_run)
    train(game, evaluator, 500000,5000,100, delta_w=0.2, delta_l=0.8)
