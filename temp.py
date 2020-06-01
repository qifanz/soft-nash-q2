import pickle
import numpy as np
from environment.soccer import SoccerEnv
from util.policy_evaluator import PolicyEvaluator

env = SoccerEnv()
ground_truth_policies_file = 'data/Soccer/nash_values.pkl'
ground_truth_values_file = 'data/Soccer/nash_policies.pkl'
evaluator = PolicyEvaluator(env, ground_truth_policies_file, ground_truth_values_file, threshold=0.03)

policy_file_good = 'debug_prior_230022.pkl'
f = open(policy_file_good, 'rb')
policy_good = pickle.load(f)
f.close()
prior_file_bad = 'debug_prior_95770.pkl'
f = open(prior_file_bad, 'rb')
prior_bad = pickle.load(f)
f.close()
policy_file_bad = 'debug_policy_95770.pkl'
f = open(policy_file_bad, 'rb')
policy_bad = pickle.load(f)
f.close()

Q_good_file = 'debug_q_89520.pkl'
Q_bad_file = 'debug_q_95770.pkl'

f = open(Q_good_file, 'rb')
q_good = pickle.load(f)
f.close()
f = open(Q_bad_file, 'rb')
q_bad = pickle.load(f)
f.close()


res_good = evaluator.validate(policy_good)
res_bad = evaluator.validate(prior_bad)
res_policy_bad = evaluator.validate(policy_bad)


for i in res_good[0]:
    if i not in res_bad[0]:
        print('state ',i)
        print('nash policy: ', evaluator.nash_policies[0][i], evaluator.nash_policies[1][i])
        print('good policy: ', policy_good[0][i], policy_good[1][i])
        print('bad policy: ', policy_bad[0][i], policy_bad[1][i])

# res = evaluator.validate(policy)
print('done')
