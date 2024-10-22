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
mcts.train(5)

mctsdnn = MCTSDNN(env, size, "Go2")
mctsdnn.train(5)
networkWins = 0
ties = 0
treeWins = 0
for i in range(0,100):
    env.reset()
    mcts.reset()
    mctsdnn.reset() 
    done = False
    while not done:
        if i % 2 == 0:
            action = mcts.take_turn_play()
            _, _, done, _ = env.step(action)
            
            mctsdnn.opponent_turn_update(action)
                    
            if done:
                break
            
            action = mctsdnn.take_turn()
            _, _, done, _ = env.step(action)
            mcts.opponent_turn_update(action)
        else:
            action = mcts.take_turn_play()
            _, _, done, _ = env.step(action)
            
            mctsdnn.opponent_turn_update(action)
                    
            if done:
                break
            
            action = mctsdnn.take_turn()
            _, _, done, _ = env.step(action)
            mcts.opponent_turn_update(action)
    
    if env.winner() == -1:
        networkWins += 1
    elif env.winner() == 0:
        ties += 1
    else:
        treeWins += 1


print("Network wins: " + str(networkWins) + " out of 100 games")
    
        
    
        
    


















