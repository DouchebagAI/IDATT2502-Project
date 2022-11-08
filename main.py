import json
import warnings
warnings.filterwarnings("ignore")
import gym

from MCTS.MCTSAdrian import MCTS
from GameManager import GameManager
go_env = gym.make('gym_go:go-v0', size=3, komi=0, reward_method='real')

#print(go_env.valid_moves())

gm = GameManager(go_env)

mctsShit = MCTS(go_env)
mctsGod = MCTS(go_env)

gm.train(mctsShit, n=1000)
mctsShit.print_tree()

gm.train(mctsGod, n=10000)
black_wins = 0
white_wins = 0
tie = 0
mctsGod.print_tree()

for i in range(100):
    print(f"Match {i}")
    val = gm.test(mctsShit, mctsGod)
    match val: 
        case 1:
            black_wins += 1
        case -1:
            white_wins += 1
        case _:
            tie += 1
 

print("\n\n****************")     
print(f"Black wins: {black_wins}")
print(f"White wins: {white_wins}")
print(f"Tie: {tie}")


    




