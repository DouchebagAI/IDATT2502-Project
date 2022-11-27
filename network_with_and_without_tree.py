import gym
import numpy as np
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)

noTree = MCTSDNN(env, size, "Go")
noTree.train(10)

withTree = MCTSDNN(env, size, "Go" )
withTree.train(10)

noTreeWins = 0
withTreeWins = 0
draws = 0

for i in range(0,100):
    print("Game: ", i)
    env.reset()
    noTree.reset()
    withTree.reset() 
    done = False
    while not done:
        # Every second game switch starting player
        if( i % 2 == 0):
            action = noTree.take_turn_play()
            _, _, done, _ = env.step(action)
            
            withTree.opponent_turn_update(action)
                    
            if done:
                break
            
            action = withTree.take_turn()
            _, _, done, _ = env.step(action)
            noTree.opponent_turn_update(action)
        else:
            action = withTree.take_turn()
            _, _, done, _ = env.step(action)
            noTree.opponent_turn_update(action)
            
            if done:
                break
            
            action = noTree.take_turn_play()
            _, _, done, _ = env.step(action)
            
            withTree.opponent_turn_update(action)
    if env.winner() == 1:
        noTreeWins += 1
    elif env.winner() == -1:
        withTreeWins += 1
    else:
        draws += 1


# Plot the results in a bar graph
objects = ('No Tree', 'With Tree', 'Draws')
y_pos = np.arange(len(objects))
performance = [noTreeWins, withTreeWins, draws]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Wins')
plt.title('Wins with and without tree')
plt.show()


    
        
    
        
    


















