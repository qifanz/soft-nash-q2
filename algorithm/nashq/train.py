import csv

from algorithm.nashq.player import *
from algorithm.nashq.q import *
from util.policy_evaluator import *


def train(game,
          evaluator,
          log_file,
          policies_file,
          is_terminal_state,
          lr=0.1,
          lr_anneal_factor=0.9,
          total_n_episodes=250001,
          evaluate_frequency=5000,
          update_frequency=20000,
          epsilon=0.2,
          verbose=False,
          with_validation=True,
          run_info='*',
          precision=4
          ):
    start = time.time()

    q = Q(game.get_n_states(), game.get_n_actions(), 0.9, lr)
    player = NashPlayer(0, q, game.get_n_actions(), epsilon)
    opponent = NashPlayer(1, q, game.get_n_actions(), epsilon)

    cumulative_rewards = []
    total_steps = 0

    episode_steps = 0
    max_q_update = 0

    next_update_episode = update_frequency
    for episode in range(total_n_episodes):
        state = game.reset()
        Q_copy = q.Q.copy()
        reward_episode = 0
        # play one episode, update Q at each step
        while True:
            episode_steps += 1
            total_steps += 1
            action_pl = player.choose_action(state)
            action_op = opponent.choose_action(state)
            new_state, reward, is_terminated, info = game.step(action_pl, action_op)
            px, py = q.update(reward, state, action_pl, action_op, new_state, precision=precision)
            player.update_policy(new_state, px)
            opponent.update_policy(new_state, py)
            state = new_state
            reward_episode += reward
            if is_terminated:
                reward_episode -= reward * 0.9  # stupid trick here, improve it
                break

        cumulative_rewards.append(reward_episode)
        current_update = np.max(np.abs(np.subtract(q.Q, Q_copy)))
        max_q_update = max(max_q_update, current_update)

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
                average_reward = np.mean(cumulative_rewards[-evaluate_frequency:])
                print('-----------------------------------------------')
                print(run_info)
                print('[episode', episode, '/', total_n_episodes, ']')
                print('     Q diff', max_q_update)
                print('     average step', episode_steps / evaluate_frequency)
                print('     average reward', average_reward)
                print("     Correct %", len(res[0]) / (len(res[0]) + len(res[1])), '[', len(res[0]), '/',
                      (len(res[0]) + len(res[1])), ']')
                print('     Deviation min', min(res[2]), 'average', np.mean(res[2]), 'median', np.median(res[2]), 'max',
                      max(res[2]))
                print('used', time.time() - start)
                write_log(episode, time.time() - start, max_q_update,
                          episode_steps / evaluate_frequency, average_reward, len(res[0]), log_file)
            episode_steps = 0
            max_q_update = 0

        if episode == next_update_episode:
            q.lr *= lr_anneal_factor
            next_update_episode += update_frequency

    policies = gen_policies(player, opponent, game, policies_file)
    if with_validation:
        res = evaluator.validate(policies)
        print("Correct %", len(res[0]) / (len(res[0]) + len(res[1])), len(res[0]), '/', (len(res[0]) + len(res[1])))
        print('     Deviation min', min(res[2]), 'average', np.mean(res[2]), 'median', np.median(res[2]), 'max',
              max(res[2]))

    print('Training used in total', time.time() - start)
    return policies, cumulative_rewards


def write_log(iteration, time, q_diff, average_step, average_reward, correct_percentage, file):
    f = open(file, 'a+', newline='')
    writer = csv.writer(f)
    writer.writerow([iteration, time, q_diff, average_step, average_reward, correct_percentage])
    f.close()


def gen_policies(pl, op, env, file):
    policies_player = pl.generate_all_policies(env.get_n_states())
    policies_op = op.generate_all_policies(env.get_n_states())
    policies = [policies_player, policies_op]
    if file is not None:
        f = open(file, 'wb')
        pickle.dump(policies, f)
        f.close()
    return policies
