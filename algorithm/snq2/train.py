import csv

from algorithm.snq2.double_Q import *
from algorithm.snq2.kl_Q import QWithKL
from algorithm.snq2.player import *
from util.policy_evaluator import *


def train(game,
          evaluator,
          log_file,
          policies_file,
          is_terminal_state,
          prior_update_factor=0,
          beta_anneal_factor=None,
          lr=0.1,
          lr_anneal_factor=0.9,
          total_n_episodes=250001,
          fixed_beta_episode=None,
          evaluate_frequency=5000,
          update_frequency=20000,
          update_frequency_ub=40000,
          update_frequency_lb=5000,
          nash_requency=10,
          beta_pl=20, beta_op=-20,
          beta_threshold=0.1,
          epsilon=0.2,
          verbose=False,
          with_validation=True,
          reference_init='uniform',
          prior_file='',
          update_schedule='dynamic',
          run_info='*'
          ):
    if fixed_beta_episode is None:
        fixed_beta_episode = int(0.7 * total_n_episodes)
    if beta_anneal_factor is None:
        beta_anneal_factor = math.pow(0.2 / beta_pl, update_frequency / fixed_beta_episode)

    start = time.time()

    reference_policy = ReferencePolicy(game.get_n_states(), game.get_n_actions(), strategy=reference_init,
                                       prior_file=prior_file)
    q_kl = QWithKL(game.get_n_states(), game.get_n_actions(), 0.9, lr)
    q = DoubleQ(game.get_n_states(), game.get_n_actions(), 0.9, lr)
    player = Player(0, q_kl, q, game.get_n_actions(), beta_pl, beta_op, reference_policy, epsilon, True)
    opponent = Player(1, q_kl, q, game.get_n_actions(), beta_op, beta_pl, reference_policy, epsilon, True)
    q.set_players(player, opponent)

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
            q_kl.update(reward, state, action_pl, action_op, new_state, player)
            use_nash_update = total_steps % nash_requency == 0
            q.update(reward, state, action_pl, action_op, new_state, player, use_nash_update)
            state = new_state
            reward_episode += reward
            if is_terminated:
                reward_episode -= reward * 0.9  # stupid trick here, improve it
                break

        cumulative_rewards.append(reward_episode)
        current_update = np.max(np.abs(np.subtract(q.Q, Q_copy)))
        max_q_update = max(max_q_update, current_update)

        epsilon_annealed = max(epsilon * (1 - 1.2 * episode / total_n_episodes), 0)
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
                if len(res[2])!=0:
                    print('     Deviation min', min(res[2]), 'average', np.mean(res[2]), 'max', max(res[2]))
                print('used', time.time() - start)
                write_log(episode, time.time() - start, max_q_update,
                          episode_steps / evaluate_frequency, average_reward, len(res[0]), log_file)
            episode_steps = 0
            max_q_update = 0

        if episode == next_update_episode:
            print('update at', episode)
            is_update_close = reference_policy.update_reference(q, prior_update_factor, is_terminal_state)
            update_frequency, q, q_kl, player, opponent = update_params(update_schedule, is_update_close,
                                                                        update_frequency, update_frequency_lb,
                                                                        update_frequency_ub, q, q_kl, player, opponent,
                                                                        lr_anneal_factor,
                                                                        beta_anneal_factor, beta_threshold)

            next_update_episode += update_frequency
            print('update frequency switched to', update_frequency)

    policies = gen_policies(player, opponent, game, policies_file)
    if with_validation:
        res = evaluator.validate(policies)
        print("Correct %", len(res[0]) / (len(res[0]) + len(res[1])), len(res[0]), '/', (len(res[0]) + len(res[1])))
        print('     Deviation min', min(res[2]), 'average', np.mean(res[2]), 'max', max(res[2]))

    print('Training used in total', time.time() - start)
    return policies, cumulative_rewards


def update_params(update_schedule, is_update_close, update_frequency, update_frequency_lb, update_frequency_ub, q, q_kl,
                  pl, op, lr_anneal_factor, beta_anneal_factor, beta_threshold):
    if update_schedule == 'dynamic':
        if not is_update_close:
            update_frequency = int(max(update_frequency * 0.75, update_frequency_lb))
        else:
            q.lr *= 0.8
            q_kl.lr *= 0.8
            pl.beta *= 0.8
            op.beta *= 0.8
            pl.beta_op *= 0.8
            op.beta_op *= 0.8
            update_frequency = int(min(update_frequency * 1.3, update_frequency_ub))
    q.lr *= lr_anneal_factor
    q_kl.lr *= lr_anneal_factor
    beta = max(pl.beta * beta_anneal_factor, beta_threshold)
    beta_op = max(op.beta * beta_anneal_factor, beta_threshold)
    pl.beta = beta
    op.beta = beta_op
    pl.beta_op = beta_op
    op.beta_op = beta
    return update_frequency, q, q_kl, pl, op


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
