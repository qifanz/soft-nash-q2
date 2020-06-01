from algorithm.singleq.train import *
from environment.PEG_game import *
from environment.det_peg import DetPEG

env = DetPEG()
game = PEGGame(env)
init_update_freq = 10000
ground_truth_policies_file = 'data/' + env.get_name() + '/nash_values.pkl'
ground_truth_values_file = 'data/' + env.get_name() + '/nash_policies.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.2
total_run = 5
for i in range(total_run):
    info = '| init_update_freq: ' + str(
        init_update_freq) + '| init_lr:' + str(init_lr) + '| No.' + str(i + 1) + '/' + str(total_run)
    train(game,
          evaluator,
          lr=init_lr,
          lr_anneal_factor=0.95,
          verbose=True,
          update_frequency=init_update_freq,
          total_n_episodes=100001,
          evaluate_frequency=2000,
          run_info=info)
