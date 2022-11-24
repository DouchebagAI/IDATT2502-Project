import gym
import numpy as np
from MCTS.MCTS import MCTS
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)


no_data_aug = MCTSDNN(env, size, "Go2",data_augment=False)
data_aug = MCTSDNN(env, size, "Go2",data_augment=True)

no_data_aug.train(5)
data_aug.train(5)
networkWins = 0
ties = 0
treeWins = 0
for i in range(0,1000):
    env.reset()
    data_aug.reset()
    no_data_aug.reset() 
    done = False
    while not done:
        action = no_data_aug.take_turn_play()
        _, _, done, _ = env.step(action)
        
        data_aug.opponent_turn_update(action)
                
        if done:
            break
        
        action = data_aug.take_turn_2()
        _, _, done, _ = env.step(action)
        no_data_aug.opponent_turn_update(action)
    
    
    if env.winner() == -1:
        networkWins += 1
    elif env.winner() == 0:
        ties += 1
    else:
        treeWins += 1


print("DataAugmentation wins: " + str(networkWins) + " out of 1000 games")