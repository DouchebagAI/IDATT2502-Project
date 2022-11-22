import gym
import numpy as np
from MCTS.MCTS import MCTS
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)

noTree = MCTS(env)
noTree.train(20)

withTree = MCTSDNN(env, size, "Go" )
withTree.train(20)

for i in range(0,100):
    env.reset()
    noTree.reset()
    withTree.reset() 
    done = False
    while not done:
        action = noTree.take_turn_play()
        _, _, done, _ = env.step(action)
        
        withTree.opponent_turn_update(action)
                
        if done:
            break
        
        action = withTree.take_turn()
        _, _, done, _ = env.step(action)
        noTree.opponent_turn_update(action)
    env.winner()
    
        
    
        
    


















