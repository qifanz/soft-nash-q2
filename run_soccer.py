import numpy as np

from environment.soccer import SoccerEnv
from environment.soccer_game import SoccerGame

game = SoccerGame(SoccerEnv())
import time

for episode in range(10):
    print('start')
    state = game.reset()
    r1, c1, r2, c2, ball = game.env._index2rcb(state)
    print(r1, c1, r2, c2, ball)
    game.render()
    time.sleep(0.5)
    while True:
        action1 = np.random.randint(5)
        action2 = np.random.randint(5)
        print(game.env.get_action_str(action1), game.env.get_action_str(action2))
        state, reward, done, _ = game.step(action1, action2)
        print('Reward ', reward)
        if state is not None:
            r1, c1, r2, c2, ball = game.env._index2rcb(state)
            print(r1, c1, r2, c2, ball)
            game.render()
            time.sleep(0.5)
        if done:
            break
