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
for i in range(5):
    log_file = 'data/' + env.get_name() + '/nashq/log' + str(init_lr) + '.csv'
    policies, cumulative_reward = train(game,
                                        evaluator,
                                        log_file,
                                        policies_file,
                                        env.is_terminal_state,  # just to make life easier
                                        lr=init_lr,
                                        lr_anneal_factor=0.8,
                                        verbose=True,
                                        update_frequency=20000,
                                        total_n_episodes=200001,
                                        evaluate_frequency=2000
                                        )
    f = open('data/' + env.get_name() + '/nashq/rewards_' + str(i) + '.pkl', 'wb')
    pickle.dump(cumulative_reward, f)
    f.close()
