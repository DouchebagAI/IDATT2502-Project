import gym
import numpy as np
from MCTS.MCTS import MCTS
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

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
    plt.ylim(100)
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + " Accuracy -" + f" Max: {np.max(accuracy)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)

playerGo = MCTSDNN(env, size, "Go", kernel_size=3)
playerGo.train(20, mse_loss=True)
playerGo2 = MCTSDNN(env, size, "Go2", kernel_size=3)
playerGo2.train(20, mse_loss=True)

plot_training(playerGo, "MSE Loss Go1", playerGo.model_losses, playerGo.model_accuracy)

plot_training(playerGo2, "MSE Loss Go2", playerGo2.model_losses, playerGo2.model_accuracy)

playerGo_win = 0
playerGo2_win = 0
draws = 0

for i in range(0,100):
    env.reset()
    playerGo.reset()
    playerGo2.reset() 
    done = False
    while not done:
        action = playerGo.take_turn_2()
        _, _, done, _ = env.step(action)
        
        playerGo2.opponent_turn_update(action)
                
        if done:
            break
        
        action = playerGo2.take_turn_2()
        _, _, done, _ = env.step(action)
        playerGo.opponent_turn_update(action)
    env.winner()
    if env.winner() == 1:
        playerGo_win += 1
    elif env.winner() == -1:
        playerGo2_win += 1
    else:
        draws +=1


# Plot the results in a bar graph
objects = ('Go', 'Go2', 'Draws')
y_pos = np.arange(len(objects))
performance = [playerGo_win, playerGo2_win, draws]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Wins')
plt.title('Wins with Go vs Go2')
plt.show()

