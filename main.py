import json
import warnings
warnings.filterwarnings("ignore")
import gym

import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager
size = 5
go_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

#print(go_env.valid_moves())

gm = GameManager(go_env)


def play(black_p, white_p, n=100):
    white = 0
    black = 0
    tie = 0
    for i in range(n):
        outcome = gm.test(black_p, white_p)
        if outcome == 1:
            black += 1
        if outcome == -1:
            white += 1
        if outcome == 0:
            tie += 1

    print("Go: black: Go2: white")
    print(f"black wins: {black} white wins: {white}  tie: {tie}")

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    langs = ['black', 'white', 'tie']
    students = [black, white, tie]
    ax.bar(langs, students)
    plt.show()

#go_env.reset()
#state, reward, done, info = go_env.step(9)
#print(state[2][0])
#print(reward)
#print(info)

player10 = MCTSDNN(go_env, size, "Go2", kernel_size=5)
print("Trener første tre")
player10.train(5)
#player10.print_tree()
player100 = MCTSDNN(go_env, size, "Go2", kernel_size=5)
print("Trener andre tre")
player100.train(10)
player3 = MCTSDNN(go_env, size, "Go2", kernel_size=3)
print("Trener andre tre")
player3.train(20)

#player100.print_tree()
#gm.play_as_white(player100)



print("5 vs 10")
play(player10, player100)
print("10 vs 5")
play(player100, player10)

print("5 vs 20")
play(player10, player3)
print("20 vs 5")
play(player3, player10)

print("20 vs 10")
play(player3, player100)
print("10 vs 20")
play(player100, player3)

#
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
        




    




