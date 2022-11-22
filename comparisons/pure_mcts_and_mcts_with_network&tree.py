import gym
import numpy as np
from MCTS.MCTS import MCTS
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)

mcts = MCTS(env)
mcts.train(20)

mctsdnn = MCTSDNN(env, size, "Go" )
mctsdnn.train(20)
win_dict = dict()
win_dict.update({mcts: 0 })
win_dict.update({mctsdnn: 0})

for i in range(0,100):
    env.reset()
    mcts.reset()
    mctsdnn.reset() 
    done = False
    while not done:
        action = mcts.take_turn_play()
        _, _, done, _ = env.step(action)
        
        mctsdnn.opponent_turn_update(action)
                
        if done:
            break
        
        action = mctsdnn.take_turn_play()
        _, _, done, _ = env.step(action)
        mcts.opponent_turn_update(action)
    if env.winner() == 1:
        win_dict.update({mcts: win_dict.get(mcts) + 1})
    elif env.winner() == -1:
         win_dict.update({mctsdnn: win_dict.get(mctsdnn) + 1})

plt.figure()
plt.title("Tournament : As model 0")
plt.xlabel("Model iterations")
plt.ylim(0,1000)
plt.ylabel("Wins for model 0")
plt.plot(np.array(list(win_dict.keys())),np.array(list(win_dict.values())))
plt.show()











