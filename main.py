import json
import warnings
warnings.filterwarnings("ignore")
import gym

from modelsV2.MCTS import MCTS
from modelsV2.GameManager import GameManager
go_env = gym.make('gym_go:go-v0', size=3, komi=0, reward_method='real')

gm = GameManager(go_env)

mctsShit = MCTS(go_env)
mctsGod = MCTS(go_env)

gm.train(mctsShit, n=100)
gm.train(mctsGod, n=10000)
mctsGod.R.check_ns()
mctsShit.R.check_ns()
black_wins = 0
white_wins = 0
tie = 0

for i in range(1000):
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


    




