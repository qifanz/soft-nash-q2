from algorithm.snq2.train import *
from environment.PEG_game import *
from environment.high_dim_PEG import HighDimEnv

env = HighDimEnv()
game = PEGGame(env)
prior_init = 'uniform'
# prior = 'quasi-nash'
ground_truth_policies_file = 'data/' + env.get_name() + '/nash_values.pkl'
ground_truth_values_file = 'data/' + env.get_name() + '/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/snq2_policies_' + prior_init + '.pkl'
prior_file = 'data/' + env.get_name() + '/prior.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
schedule = 'dynamic'

for init_update_freq in [30000]:
      log_file = 'data/' + \
                 env.get_name() + \
                 '/snq2/log_' + \
                 prior_init + '_' + \
                 schedule + '_' + \
                 str(init_update_freq) + '.csv'
      train(game,
            evaluator,
            log_file,
            policies_file,
            env.is_terminal_state,  # just to make life easier
            lr=0.2,
            lr_anneal_factor=0.95,
            verbose=True,
            beta_op=-50, beta_pl=50,
            update_frequency=init_update_freq,
            update_frequency_ub=60000,
            update_frequency_lb=15000,
            prior_update_factor=0,
            total_n_episodes=600001, fixed_beta_episode=500000,
            reference_init=prior_init,
            prior_file=prior_file,
            update_schedule=schedule)
