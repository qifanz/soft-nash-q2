from algorithm.nashq.train import *
from environment.PEG_game import *
from environment.low_dim_PEG import *

env = LowDimPEG()
game = PEGGame(env)
prior = 'uniform'
ground_truth_policies_file = 'data/LowDimensionPEG/nash_values.pkl'
ground_truth_values_file = 'data/LowDimensionPEG/nash_policies.pkl'
policies_file = 'data/' + env.get_name() + '/nashq_policies.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file)
init_lr = 0.1
lr_anneal_factor = 0.8
total_runs = 5
for i in range(total_runs):
    log_file = 'data/' + env.get_name() + '/nashq/log' + str(init_lr) + '.csv'
    reward_file = 'data/' + env.get_name() + '/nashq/reward' + str(init_lr) + '.pkl'
    info = 'lr:' + str(init_lr) + ' |lr_anneal_factor:' + str(lr_anneal_factor) + 'No.' + str(i) + '/' + str(total_runs)
    policies, cumulative_reward = train(game,
                                        evaluator,
                                        log_file,
                                        policies_file,
                                        env.is_non_terminal_state,  # just to make life easier
                                        lr=init_lr,
                                        lr_anneal_factor=lr_anneal_factor,
                                        verbose=True,
                                        update_frequency=10000,
                                        total_n_episodes=300001,
                                        evaluate_frequency=2000,
                                        run_info=info,

                                        )
    f = open(reward_file, 'wb')
    pickle.dump(cumulative_reward, f)
    f.close()
