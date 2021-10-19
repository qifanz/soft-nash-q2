from algorithm.snq2.train import *
from environment.PEG_game import *
from environment.high_dim_PEG import HighDimEnv

env = HighDimEnv()
game = PEGGame(env)
prior_init = 'quasi-nash'
# prior = 'quasi-nash'
ground_truth_policies_file = 'data/' + env.get_name() + '/nash_values.pkl'
ground_truth_values_file = 'data/' + env.get_name() + '/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/snq2_policies_' + prior_init + '.pkl'
prior_file = 'data/' + env.get_name() + '/prior.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
schedule = 'dynamic'
init_lr = 0.2
total_run = 5
for i in range(total_run):
    for init_update_freq in [25000]:
        log_file = 'data/' + \
                   env.get_name() + \
                   '/snq2/log_' + \
                   prior_init + '_' + \
                   schedule + '_' + \
                   str(init_update_freq) + '_' + str(init_lr) + '.csv'
        rewards_file = 'data/' + \
                       env.get_name() + \
                       '/snq2/reward_' + \
                       prior_init + '_' + \
                       schedule + '_' + \
                       str(init_update_freq) + '_' + str(init_lr) + '_' + str(i) + '.pkl'
        info = 'prior:' + prior_init + '| schedule:' + schedule + '| init_update_freq: ' + str(
            init_update_freq) + '| init_lr:' + str(init_lr) + '| No.' + str(i + 1) + '/' + str(total_run)
        policies, cumulative_reward = train(game,
                                            evaluator,
                                            log_file,
                                            policies_file,
                                            env.is_non_terminal_state,  # just to make life easier
                                            lr=0.2,
                                            lr_anneal_factor=0.9,
                                            verbose=True,
                                            beta_op=-25, beta_pl=25,
                                            update_frequency=init_update_freq,
                                            update_frequency_ub=init_update_freq * 1.5,
                                            update_frequency_lb=int(init_update_freq / 2),
                                            prior_update_factor=0.7,
                                            total_n_episodes=600001, fixed_beta_episode=500000,
                                            reference_init=prior_init,
                                            prior_file=prior_file,
                                            update_schedule=schedule,
                                            run_info=info)
        f = open(rewards_file, 'wb')
        pickle.dump(cumulative_reward, f)
        f.close()
