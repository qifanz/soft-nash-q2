from algorithm.nashq.train import *
from environment.PEG_game import *
from environment.high_dim_PEG import HighDimEnv

env = HighDimEnv()
game = PEGGame(env)

ground_truth_policies_file = 'data/' + env.get_name() + '/nash_values.pkl'
ground_truth_values_file = 'data/' + env.get_name() + '/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/nashq_policies.pkl'
prior_file = 'data/' + env.get_name() + '/prior.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
schedule = 'dynamic'
log_file = 'data/' + env.get_name() + '/nashq/log.csv'
train(game,
      evaluator,
      log_file,
      policies_file,
      env.is_non_terminal_state,  # just to make life easier
      lr=0.2,
      lr_anneal_factor=0.95,
      verbose=True,
      total_n_episodes=400001, )
