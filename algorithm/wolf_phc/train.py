from algorithm.wolf_phc.player import Player


def train(game,
          evaluator,
          total_n_episodes,
          evaluate_frequency=1000,
          update_frequency=10000):
    player = Player(0, game.get_n_states(), game.get_n_actions(), 0.1, 0.4, 0.1)
    opponent = Player(1, game.get_n_states(), game.get_n_actions(), 0.1, 0.4, 0.1)

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
        if episode % update_frequency == 0 and episode!=0:
            player.lr /= (episode/update_frequency)
            opponent.lr /= (episode/update_frequency)
            player.delta_l /= (episode/update_frequency)
            player.delta_w /= (episode/update_frequency)
            opponent.delta_l /= (episode/update_frequency)
            opponent.delta_w /= (episode/update_frequency)

def gen_policies(player, opponent):
    return [player.get_all_policies(), opponent.get_all_policies()]