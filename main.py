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

def plot_training(mcts: MCTSDNN, title, loss, acc):
    losses = np.array(loss)
    accuracy = np.array(acc)
    plt.figure()
    plt.plot(losses, label="Loss", color = "blue")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title+ f" Min: {np.min(losses)}")
    plt.show()
    #plt.savefig(f"graphs/{uuid.uuid4()}.png")

    plt.figure()
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + f" Max: {np.max(accuracy)}")
    plt.show()
    #plt.savefig(f"graphs/{uuid.uuid4()}.png")


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
    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    langs = ['black', 'white', 'tie']
    students = [black, white, tie]
    ax.bar(langs, students)
    plt.show()
    """

#go_env.reset()
#state, reward, done, info = go_env.step(9)
#print(state[2][0])
#print(reward)
#print(info)
player_tree_only = MCTSTREE(go_env)
print("Trener standard MCTS")
#player_tree_only.train(5)

#player_tree_only2 = MCTSTREE(go_env)
#player_tree_only2.train(2)
#player10 = MCTSDNN(go_env, size, "Go", kernel_size=5)

#player10.print_tree()
playerCNN = MCTSDNN(go_env, size, "Go2", kernel_size=3)
#print("Trener andre tre")
print("Trener MCTS /m CNN")





#gm.play_tournament()
playerCNN.train(10)
player_tree_only.train(5)
print("CNN vs Tree")
play(playerCNN, player_tree_only, 1000)
print("Tree vs CNN")
play(player_tree_only, playerCNN, 1000)

plot_training(playerCNN, "Model 1", playerCNN.model_losses, playerCNN.model_accuracy)
plot_training(playerCNN, "Model 2", playerCNN.value_model_losses, playerCNN.value_model_accuracy)
    




