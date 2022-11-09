import json
import warnings
warnings.filterwarnings("ignore")
import gym

from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager
go_env = gym.make('gym_go:go-v0', size=3, komi=0, reward_method='real')

go_env.__sizeof__()
#print(go_env.valid_moves())

gm = GameManager(go_env)
#go_env.reset()
#state, reward, done, info = go_env.step(9)
#print(state)
#print(reward)
#print(info)

player10 = MCTSDNN(go_env)
player10.train(3)
#player10.print_tree()
player100 = MCTSDNN(go_env)
player100.train(20)
#player100.print_tree()

best_wins = 0
other_wins = 0
tie = 0
for i in range(100):
    outcome = gm.test(player10, player100)
    match outcome: 
            case 1:
                other_wins += 1
            case -1:
                best_wins += 1
            case _:
                tie += 1
print(f"white wins: {best_wins} black wins: {other_wins} tie: {tie}")


"""""""""
for i in range(5): 
    mcts = MCTSDNN(go_env)
    mcts.train_simulate
    mctsIterations.append(mcts)

for index in range(len(mctsIterations)-1):
    best_wins = 0
    other_wins = 0
    tie = 0
    for _ in range(10):
        outcome = gm.test(mctsIterations[index], mctsIterations[len(mctsIterations)-1])
        match outcome: 
            case 1:
                other_wins += 1
            case -1:
                best_wins += 1
            case _:
                tie += 1
        outcome = gm.test(mctsIterations[len(mctsIterations)-1], mctsIterations[index])
        match outcome: 
            case 1:
                best_wins += 1
            case -1:
                other_wins += 1
            case _:
                tie += 1 
    print(f"round: {index}, "f"Best wins: {best_wins}", f"Other wins: {other_wins}", f"Tie: {tie}")
        
        """
        




    




