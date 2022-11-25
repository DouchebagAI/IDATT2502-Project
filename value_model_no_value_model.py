import gym
import numpy as np
from MCTS.MCTS import MCTS
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager
import time

def plot_training(mcts: MCTSDNN, title, loss, acc):
    losses = np.array(loss)
    accuracy = np.array(acc)
    plt.figure()
    plt.ylim(0, 10)
    plt.plot(losses, label="Loss", color="blue")
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title(title + " Loss -" f" Min: {np.min(losses)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

    plt.figure()
    plt.ylim(0, 1)
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + " Accuracy -" + f" Max: {np.max(accuracy)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")


size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)




no_val_net = MCTSDNN(env, size, "Go2", kernel_size=3, value_network=False)

start_time_train_no_val = time.time()
no_val_net.train(10, mse_loss=True)
total_time_train_no_val = time.time() - start_time_train_no_val

val_net = MCTSDNN(env, size, "Go2", kernel_size=3, value_network=True)

start_time_train_val = time.time()
val_net.train(10, mse_loss=True)
total_time_train_val = time.time() - start_time_train_val

plot_training(no_val_net, f"MSE Loss No Value model - Training time: {total_time_train_no_val}", no_val_net.model_losses, no_val_net.model_accuracy)

plot_training(val_net, f"MSE Loss Value model - Training time: {total_time_train_val}", val_net.model_losses, val_net.model_accuracy)

playerGo_win = 0
playerGo2_win = 0
draws = 0

time_pr_round_no_val = []
time_pr_round_val = []

for i in range(0, 100):
    print(i)
    env.reset()
    no_val_net.reset()
    val_net.reset()
    done = False
    while not done:
        start_time = time.time()
        action = no_val_net.take_turn_2()
        time_pr_round_no_val.append(time.time() - start_time)

        _, _, done, _ = env.step(action)

        val_net.opponent_turn_update(action)

        if done:
            break

        start_time = time.time()
        action = val_net.take_turn_2()
        time_pr_round_val.append(time.time() - start_time)

        _, _, done, _ = env.step(action)
        no_val_net.opponent_turn_update(action)
    env.winner()
    no_val_net.move_count = 0
    val_net.move_count = 0
    if env.winner() == 1:
        playerGo_win += 1
    elif env.winner() == -1:
        playerGo2_win += 1
    else:
        draws += 1

time = np.array(time_pr_round_no_val)
plt.figure()
plt.plot(time, label="Time per round", color="blue")
plt.xlabel('Rounds')
plt.ylabel('Seconds')
plt.title(f"Time per round (No Value Network) - Average: {np.mean(time)}")
plt.show()

time1 = np.array(time_pr_round_val)
plt.figure()
plt.plot(time1, label="Time per round", color="green")
plt.xlabel('Rounds')
plt.ylabel('Seconds')
plt.title(f"Time per round (Value Network) - Average: {np.mean(time1)}")
plt.show()

plt.figure()
plt.plot(time, label="No Value Network", color="blue")
plt.plot(time1, label="Value Network", color="green")
plt.xlabel('Rounds')
plt.ylabel('Seconds')
plt.title(f"Time per round - Average: {np.mean(time1)}")
plt.show()

# Plot the results in a bar graph
objects = ('No Value Network (black)', 'Value Network (white)', 'Draws')
y_pos = np.arange(len(objects))
performance = [playerGo_win, playerGo2_win, draws]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Wins')
plt.title('Wins No Value Network vs Value Network')
plt.show()

