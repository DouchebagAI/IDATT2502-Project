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
        
        action = mctsdnn.take_turn()
        _, _, done, _ = env.step(action)
        mcts.opponent_turn_update(action)
    env.winner()
    
        
    
        
    


















