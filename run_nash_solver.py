from algorithm.modified_newton.nash_solver import *
from environment.low_dim_PEG import *

low_dim_env = LowDimPEG()
nash_value_file = 'data/' + low_dim_env.get_name() + '/nash_values.pkl'
nash_policy_file = 'data/' + low_dim_env.get_name() + '/nash_policies.pkl'
nash_solver = NashSolver(low_dim_env, nash_value_file, nash_policy_file)
nash_solver.solve()
