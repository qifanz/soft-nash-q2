from algorithm.snq2.train import *
from environment.srps import RPSEnv

game = RPSEnv()
init_update_freq = 25
prior_init = 'quasi-uniform'
ground_truth_policies_file = None
ground_truth_values_file = None
policies_file = 'data/' + game.get_name() + '/snq2_policies' + '.pkl'
evaluator = PolicyEvaluator(game, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.5
total_run = 5
for i in range(total_run):
    log_file = 'data/' + \
               game.get_name() + \
               '/snq2/log_' + prior_init + \
               str(init_update_freq) + '_' + str(init_lr) + '.csv'
    rewards_file = 'data/' + \
                   game.get_name() + \
                   '/snq2/reward_' + \
                   str(init_update_freq) + '_' + str(init_lr) + '_' + str(i) + '.pkl'
    info = '| init_lr:' + str(init_lr) + '| No.' + str(i + 1) + '/' + str(total_run)
    policies, cumulative_reward = train(game,
                                        evaluator,
                                        log_file,
                                        policies_file,
                                        game.is_non_terminal_state,  # just to make life easier
                                        lr=init_lr,
                                        lr_anneal_factor=0.995,
                                        verbose=True,
                                        beta_op=-0.5, beta_pl=0.5,
                                        update_frequency=init_update_freq,
                                        update_frequency_ub=init_update_freq * 2,
                                        update_frequency_lb=init_update_freq / 2,
                                        prior_update_factor=1,
                                        total_n_episodes=301, fixed_beta_episode=250,
                                        evaluate_frequency=2,
                                        reference_init=prior_init,
                                        update_schedule='fixed',
                                        run_info=info,
                                        epsilon=0.5,
                                        precision=4,
                                        nash_frequency=5)
    f = open(rewards_file, 'wb')
    pickle.dump(cumulative_reward, f)
    f.close()
