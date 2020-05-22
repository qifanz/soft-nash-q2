from algorithm.snq2.train import *
from environment.PEG_game import *
from environment.low_dim_PEG import *

env = LowDimPEG()
game = PEGGame(env)
prior_init = 'uniform'
init_update_freq = 20000
# prior = 'quasi-nash'
ground_truth_policies_file = 'data/LowDimensionPEG/nash_values.pkl'
ground_truth_values_file = 'data/LowDimensionPEG/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/snq2_policies_' + prior_init + '.pkl'
prior_file = 'data/' + env.get_name() + '/prior.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.1
for schedule in ['dynamic', 'fixed']:
    for i in range(5):
        log_file = 'data/' + \
                   env.get_name() + \
                   '/snq2/log_' + \
                   prior_init + '_' + \
                   schedule + '_' + \
                   str(init_update_freq) + '_' + str(init_lr) + '.csv'

        policies, cumulative_reward = train(game,
                                            evaluator,
                                            log_file,
                                            policies_file,
                                            env.is_terminal_state,  # just to make life easier
                                            lr=init_lr,
                                            lr_anneal_factor=0.8,
                                            verbose=True,
                                            beta_op=-20, beta_pl=20,
                                            update_frequency=init_update_freq,
                                            update_frequency_ub=40000,
                                            update_frequency_lb=5000,
                                            prior_update_factor=0,
                                            total_n_episodes=200001, fixed_beta_episode=160000,
                                            evaluate_frequency=1000,
                                            reference_init=prior_init,
                                            prior_file=prior_file,
                                            update_schedule=schedule)
        f = open('data/' + env.get_name() + '/snq2/rewards_' + prior_init + '_' + schedule + '_' + str(i) + '.pkl',
                 'wb')
        pickle.dump(cumulative_reward, f)
        f.close()
