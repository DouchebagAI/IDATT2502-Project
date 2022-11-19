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
playerDCNN = MCTSDNN(go_env, size, "Go2", kernel_size=3)
playerCNN = MCTSDNN(go_env, size, "Go", kernel_size=3)
#print("Trener andre tre")
print("Trener MCTS /m CNN")
gm.play_tournament()
#print(np.load("models/training_data/value_model.npy", allow_pickle=True))
m1 = np.load("models/training_data/model_168_3e15f1db-a1a3-4b00-a262-977aecaa85e6.npy", allow_pickle=True)
m2 = np.load("models/training_data/model_e2c6149d-4745-455a-b5fc-e5383a153079.npy", allow_pickle=True)
t1 = np.load("models/test_data/model_71_44bb6ddf-46e6-4c81-bee5-2c38f2834359.npy", allow_pickle=True)
m3 = np.concatenate((m1,m2))

v_m1 = np.load("models/training_data/value_model_2616_35e8d926-b416-444b-8ef8-07eb56c975f5.npy", allow_pickle=True)
v_m2 = np.load("models/training_data/value_model_d89f8f41-cd35-4ea3-b936-a15b5b637d35.npy", allow_pickle=True)
v_t1 = np.load("models/test_data/value_model_683_d01d7120-afdd-46c5-a050-f2fc272c7f40.npy", allow_pickle=True)
v_m3 = np.concatenate((v_m2,v_m1))

playerCNN.train_model(playerCNN.model,
                            playerCNN.get_training_data(m3, t1), playerCNN.model_losses, playerCNN.model_accuracy)
playerDCNN.train_model(playerDCNN.model,
                       playerDCNN.get_training_data(m3, t1), playerDCNN.model_accuracy, playerDCNN.model_accuracy)


playerCNN.train_model_value(playerCNN.value_model,
                            playerCNN.get_training_data(v_m3, v_t1), playerCNN.value_model_losses, playerCNN.value_model_accuracy)
playerDCNN.train_model_value(playerDCNN.value_model,
                            playerDCNN.get_training_data(v_m3, v_t1), playerDCNN.value_model_losses, playerDCNN.value_model_accuracy)
#playerDCNN.train_model_value(playerDCNN.value_model,
                       #playerDCNN.get_training_data(v_m3, v_t1), playerDCNN.value_model_accuracy, playerDCNN.value_model_accuracy)


#playerCNN.training_data = m3.tolist()
#playerDCNN.training_data = m3.tolist()
#playerDCNN.training_data = m3.tolist()




#playerCNN.train(20)
#playerDCNN.train(20)
#player_tree_only.train(20)

print("CNN vs Tree")
play(playerCNN, player_tree_only, 1000)
print("Tree vs CNN")
play(player_tree_only, playerCNN, 1000)
print("CNN vs Tree")
play(playerDCNN, player_tree_only, 1000)
print("Tree vs CNN")
play(player_tree_only, playerDCNN, 1000)
play(playerDCNN, playerCNN, 1000)
print("Tree vs CNN")
play(playerCNN, playerDCNN, 1000)

plot_training(playerCNN, "Model 1", playerCNN.model_losses, playerCNN.model_accuracy)
#plot_training(playerDCNN, "Model DCNN", playerDCNN.model_losses, playerDCNN.model_accuracy)
plot_training(playerDCNN, "Model 2", playerDCNN.value_model_losses, playerDCNN.value_model_accuracy)
    




