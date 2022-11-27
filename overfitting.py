import json
import warnings
import uuid

import torch

warnings.filterwarnings("ignore")
import gym
import numpy as np
from MCTS.MCTS import MCTS as MCTSTREE
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

size = 5
go_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

go_env.reset()
m1 = np.load("models/training_data/model_168_3e15f1db-a1a3-4b00-a262-977aecaa85e6.npy", allow_pickle=True)
playerDCNN = MCTSDNN(go_env, size, "Go2", data_augment=True)
playerCNN = MCTSDNN(go_env, size, "Go", kernel_size=3)
state, reward, done, info = go_env.step(9)

print(playerDCNN.data_augmentation(m1[1]))

playerDCNN.training_data.extend(playerDCNN.data_augmentation(m1[1]))
print(playerDCNN.training_data)
gm = GameManager(go_env)


def plot_training(mcts: MCTSDNN, title, loss, acc):
    losses = np.array(loss)
    accuracy = np.array(acc)
    plt.figure()
    plt.plot(losses, label="Loss", color="blue")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title + f" Min: {np.min(losses)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

    plt.figure()
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + f" Max: {np.max(accuracy)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")


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



print(int(state[2][0][0]))
# print(reward)
# print(info)
player_tree_only = MCTSTREE(go_env)
print("Trener standard MCTS")
# player_tree_only.train(5)

# player_tree_only2 = MCTSTREE(go_env)
# player_tree_only2.train(2)
# player10 = MCTSDNN(go_env, size, "Go", kernel_size=5)

# player10.print_tree()

# print("Trener andre tre")
#print("Trener MCTS /m CNN")
#gm.play_tournament()
# print(np.load("models/training_data/value_model.npy", allow_pickle=True))

m2 = np.load("models/training_data/model_e2c6149d-4745-455a-b5fc-e5383a153079.npy", allow_pickle=True)
t1 = np.load("models/test_data/model_71_44bb6ddf-46e6-4c81-bee5-2c38f2834359.npy", allow_pickle=True)
t2 = np.load("models/test_data/model_165_dd4175f2-8059-40e3-b7e2-09ce1e772318.npy", allow_pickle=True)
m3 = np.concatenate((m1, m2))
t3 = np.concatenate((t1,t2))



#playerCNN.train_model(playerCNN.model,
                      #playerCNN.get_training_data(m3, t1), playerCNN.model_losses, playerCNN.model_accuracy)
#playerDCNN.train(4)
#playerDCNN.train_model(playerDCNN.model,
                       #playerDCNN.get_training_data(m3, t3), playerDCNN.model_losses, playerDCNN.model_accuracy)

playerDCNN.train(3)


losses = np.array(playerDCNN.model_losses)
test_losses = np.array(playerDCNN.model_accuracy)
plt.figure()
plt.plot(losses, label="Training Loss", color="blue")
plt.plot(test_losses, label = "Test loss", color="green")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title(f"Loss -  Training Min: {np.min(losses)} - Test Min: {np.min(test_losses)}")
plt.show()







