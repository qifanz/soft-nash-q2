import csv

from algorithm.singleq.Player import SingleQPlayer
from algorithm.singleq.Q import Q_table
from util.policy_evaluator import *
import time

def train(game,
          evaluator,
          lr=0.1,
          lr_anneal_factor=0.9,
          total_n_episodes=250001,
          evaluate_frequency=5000,
          update_frequency=20000,
          epsilon=0.2,
          verbose=False,
          with_validation=True,
          run_info='*',
          ):
    start = time.time()

    qpl = Q_table(game.get_n_states(), game.get_n_actions(), 0.9, lr)
    qop = Q_table(game.get_n_states(), game.get_n_actions(), 0.9, lr)
    player = SingleQPlayer(qpl, game.get_n_actions(), epsilon)
    opponent = SingleQPlayer(qop, game.get_n_actions(), epsilon)
    next_update_episode = update_frequency
    total_steps=0
    for episode in range(total_n_episodes):
        state = game.reset()

        # play one episode, update Q at each step
        while True:
            total_steps += 1
            action_pl = player.choose_action(state)
            action_op = opponent.choose_action(state)
            new_state, reward, is_terminated, info = game.step(action_pl, action_op)
            player.observe(reward, state, action_pl, action_op, new_state)
            opponent.observe(reward, state, action_op, action_pl, new_state)
            state = new_state
            if is_terminated:
                break

        epsilon_annealed = max(epsilon * (1 - 1.1 * episode / total_n_episodes), 0)
        player.epsilon = epsilon_annealed
        opponent.epsilon = epsilon_annealed

        if episode % evaluate_frequency == 0 and episode != 0:
            if with_validation:
                valid_start = time.time()
                policies = gen_policies(player, opponent, game, None)
                res = evaluator.validate(policies)
                start += (time.time() - valid_start)  # recompense evaluation time
            if verbose:
                print('-----------------------------------------------')
                print(run_info)
                print('[episode', episode, '/', total_n_episodes, ']')
                print("     Correct %", len(res[0]) / (len(res[0]) + len(res[1])), '[', len(res[0]), '/',
                      (len(res[0]) + len(res[1])), ']')
                print('     Deviation min', min(res[2]), 'average', np.mean(res[2]), 'median', np.median(res[2]), 'max',
                      max(res[2]))
                print('used', time.time() - start)


        if episode == next_update_episode:
            qpl.lr *= lr_anneal_factor
            qop.lr *= lr_anneal_factor

            next_update_episode += update_frequency


    print('Training used in total', time.time() - start)




def gen_policies(pl, op, env, file):
    policies_player = pl.generate_all_policies(env.get_n_states())
    policies_op = op.generate_all_policies(env.get_n_states())
    policies = [policies_player, policies_op]
    if file is not None:
        f = open(file, 'wb')
        pickle.dump(policies, f)
        f.close()
    return policies
