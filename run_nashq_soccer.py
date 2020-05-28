from algorithm.nashq.train import *
from environment.PEG_game import *
from environment.low_dim_PEG import *
from environment.soccer import SoccerEnv
from environment.soccer_game import SoccerGame

env = SoccerEnv()
game = SoccerGame(env)
init_update_freq = 4000
# prior = 'quasi-nash'
ground_truth_policies_file = 'data/'+env.get_name()+'/nash_values.pkl'
ground_truth_values_file = 'data/'+env.get_name()+'/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/nashq_policies_' + '.pkl'
prior_file = 'data/' + env.get_name() + '/prior.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.2
total_run = 5
for i in range(total_run):
    log_file = 'data/' + \
               env.get_name() + \
               '/nashq/log_' + \
               str(init_update_freq) + '_' + str(init_lr) + '.csv'
    rewards_file = 'data/' + \
                   env.get_name() + \
                   '/nashq/reward_' + \
                   str(init_update_freq) + '_' + str(init_lr) + '_' + str(i) + '.pkl'
    info = '| init_lr:' + str(init_lr) + '| No.' + str(i + 1) + '/' + str(total_run)
    policies, cumulative_reward = train(game,
                                        evaluator,
                                        log_file,
                                        policies_file,
                                        env.is_terminal_state,  # just to make life easier
                                        lr=init_lr,
                                        lr_anneal_factor=0.9,
                                        verbose=True,
                                        update_frequency=init_update_freq,
                                        total_n_episodes=300001,
                                        evaluate_frequency=1000,
                                        precision=4,
                                        run_info=info)
    f = open(rewards_file, 'wb')
    pickle.dump(cumulative_reward, f)
    f.close()
