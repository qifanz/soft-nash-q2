from algorithm.snq2.train import *
from environment.soccer import SoccerEnv
from environment.soccer_game import SoccerGame

env = SoccerEnv()
game = SoccerGame(env)
prior_init = 'uniform'
# prior = 'quasi-nash'
ground_truth_policies_file = 'data/' + env.get_name() + '/nash_values.pkl'
ground_truth_values_file = 'data/' + env.get_name() + '/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/snq2_policies_' + prior_init + '.pkl'
prior_file = 'data/' + env.get_name() + '/prior.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.1
total_run = 4
for schedule in ['dynamic']:
    for init_update_freq in [10000]:
        for i in range(total_run):
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
                                                lr=init_lr,
                                                lr_anneal_factor=0.99,
                                                verbose=True,
                                                beta_op=-10, beta_pl=10,
                                                update_frequency=init_update_freq,
                                                update_frequency_ub=init_update_freq * 1.5,
                                                update_frequency_lb=init_update_freq / 2,
                                                prior_update_factor=0.5,
                                                total_n_episodes=300001, fixed_beta_episode=100000,
                                                evaluate_frequency=1000,
                                                reference_init=prior_init,
                                                prior_file=prior_file,
                                                update_schedule=schedule,
                                                run_info=info,
                                                epsilon=0.7,
                                                precision=3)
            f = open(rewards_file, 'wb')
            pickle.dump(cumulative_reward, f)
            f.close()
