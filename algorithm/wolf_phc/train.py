from algorithm.wolf_phc.player import Player
import numpy as np

def train(game,
          evaluator,
          total_n_episodes,
          evaluate_frequency=1000,
          update_frequency=100,
          delta_w=0.01,
          delta_l=0.04,
          lr=0.2):
    player = Player(0, game.get_n_states(), game.get_n_actions(), delta_w, delta_l, lr,update_frequency)
    opponent = Player(1, game.get_n_states(), game.get_n_actions(), delta_w, delta_l, lr,update_frequency)

    for episode in range(total_n_episodes):
        state = game.reset()
        # play one episode, update Q at each step
        while True:
            action_pl = player.choose_action(state)
            action_op = opponent.choose_action(state)
            new_state, reward, is_terminated, info = game.step(action_pl, action_op)
            player.observe(state, action_pl, reward, new_state)
            opponent.observe(state, action_op, -reward, new_state)
            state = new_state
            if is_terminated:
                break

        if episode % evaluate_frequency == 0 and episode != 0:
            policies = gen_policies(player, opponent)
            res = evaluator.validate(policies)

            print('-----------------------------------------------')
            print('[episode', episode, '/', total_n_episodes, ']')
            print("     Correct %", len(res[0]) / (len(res[0]) + len(res[1])), '[', len(res[0]), '/',
                  (len(res[0]) + len(res[1])), ']')
            print('     Deviation min', min(res[2]), 'average', np.mean(res[2]), 'median', np.median(res[2]), 'max',
                  max(res[2]))
        #if episode % update_frequency == 0 and episode != 0:
            #player.lr /= 5
            #opponent.lr /= 5
            #player.delta_l /= 5
            #player.delta_w /= 5
            #opponent.delta_l /= 5
            #opponent.delta_w /= 5


def gen_policies(player, opponent):
    return [player.get_all_policies(), opponent.get_all_policies()]
