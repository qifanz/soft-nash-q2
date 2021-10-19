import numpy as np

from environment.soccer import SoccerEnv
from environment.soccer_game import SoccerGame

game = SoccerGame(SoccerEnv())
import time
import pickle

def import_nash_policies():
    with open('data/soccer/nash_policies.pkl','rb') as f:
        nash_policies = pickle.load(f)
    return nash_policies
step=0
total_runs=200
for episode in range(total_runs):
    #print('start')
    nash_policies = import_nash_policies()
    state = game.reset()
    r1, c1, r2, c2, ball = game.env._index2rcb(state)
    #print(r1, c1, r2, c2, ball)
    #game.render()
    #time.sleep(0.5)
    while True:
        step+=1
        if game.env.is_terminal_state(state):
            action1=0
            action2=0
        else:
            action1 = np.random.choice(np.arange(5),p=nash_policies[state][0])
            action2 = np.random.choice(np.arange(5),p=nash_policies[state][1])
        #print(game.env.get_action_str(action1), game.env.get_action_str(action2))
        state, reward, done, _ = game.step(action1, action2)
        #print('Reward ', reward)
        if state is not None:
            r1, c1, r2, c2, ball = game.env._index2rcb(state)
            #print(r1, c1, r2, c2, ball)
            #game.render()
            #time.sleep(0.5)
        if done:
            break
print(step/total_runs)
