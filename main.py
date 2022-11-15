import json
import warnings
import uuid
warnings.filterwarnings("ignore")
import gym
import numpy as np
from MCTS.MCTS import MCTS as MCTSTREE
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager
size = 5
go_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

#print(go_env.valid_moves())

gm = GameManager(go_env)

def plot_training(mcts: MCTSDNN, title):
    losses = np.array(mcts.losses)
    accuracy = np.array(mcts.accuracy)
    plt.figure(0)
    plt.plot(losses, label="Loss", color = "blue")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title+ f" Min: {np.min(losses)}")
    plt.savefig(f"graphs/{uuid.uuid4()}.png")

    plt.figure(1)
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + f" Max: {np.max(accuracy)}")
    plt.savefig(f"graphs/{uuid.uuid4()}.png")


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
player_tree_only = MCTSTREE(go_env)
#player_tree_only.train(2)

player_tree_only2 = MCTSTREE(go_env)
#player_tree_only2.train(2)
#player10 = MCTSDNN(go_env, size, "Go", kernel_size=5)
print("Go")

#player10.print_tree()
player100 = MCTSDNN(go_env, size, "Go3", kernel_size=5)
#print("Trener andre tre")
player100.train(2)
#player3 = MCTSDNN(go_env, size, "Go3", kernel_size=5)
#player100.train(5)

#plot_training(player100, "Go2 - 20 rounds training")
#player10.train(10)

print("Go2")
#player3.train(10)
#plot_training(player3, "Go3 - 20 rounds training")
#plot_training(player3, "Go3 - 5 rounds training")
#player3.train(5)

print("Go2")
#player10.train(10)
#plot_training(player3, "Go1 - 20 rounds training")
#player100.print_tree()
#gm.play_as_white(player100)


print("CNN vs Tree")
play(player_tree_only, player_tree_only2, 1000)
print("Tree vs CNN")
play(player_tree_only2, player_tree_only, 1000)




"""
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
"""
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
        




    




