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
    plt.ylim(0,10)
    plt.plot(losses, label="Loss", color="blue")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title + " Loss -" f" Min: {np.min(losses)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

    plt.figure()
    plt.ylim(0,100)
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + " Accuracy -" + f" Max: {np.max(accuracy)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

model_cross_entropy_loss = MCTSDNN(go_env, size, "Go2", kernel_size=3)
model_mse_loss = MCTSDNN(go_env, size, "Go2", kernel_size=3)

model_cross_entropy_loss.train(5, mse_loss=False)
model_mse_loss.train(5, mse_loss=True)

plot_training(model_cross_entropy_loss, "Cross Entropy Loss", model_cross_entropy_loss.model_losses, model_cross_entropy_loss.model_accuracy)

plot_training(model_mse_loss, "MSE Loss", model_mse_loss.model_losses, model_mse_loss.model_accuracy)
#plot_training(playerCNN, "Model 1", playerCNN.model_losses, playerCNN.model_accuracy)