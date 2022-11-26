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
    plt.ylim(0,1)
    plt.plot(accuracy, label="Accuracy", color="Green")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title(title + " Accuracy -" + f" Max: {np.max(accuracy)}")
    plt.show()
    # plt.savefig(f"graphs/{uuid.uuid4()}.png")

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)

playerGo = MCTSDNN(env, size, "Go", kernel_size=3, data_augment=False)
#playerGo.train(10, mse_loss=True)
playerGo2 = MCTSDNN(env, size, "Go2", kernel_size=3, data_augment=False)
#playerGo2.train(10, mse_loss=True)
m1 = np.load("models/training_data/model_168_3e15f1db-a1a3-4b00-a262-977aecaa85e6.npy", allow_pickle=True)
m2 = np.load("models/training_data/model_e2c6149d-4745-455a-b5fc-e5383a153079.npy", allow_pickle=True)
t1 = np.load("models/test_data/model_71_44bb6ddf-46e6-4c81-bee5-2c38f2834359.npy", allow_pickle=True)
m3 = np.concatenate((m1, m2))

playerGo.train_model(playerGo.model,
                       playerGo.get_training_data(m3, t1), playerGo.model_losses, playerGo.model_accuracy)

playerGo2.train_model(playerGo2.model,
                       playerGo2.get_training_data(m3, t1), playerGo2.model_losses, playerGo2.model_accuracy)

plot_training(playerGo, "MSE Loss Go1", playerGo.model_losses, playerGo.model_accuracy)

plot_training(playerGo2, "MSE Loss Go2", playerGo2.model_losses, playerGo2.model_accuracy)

playerGo_win = 0
playerGo2_win = 0
draws = 0

for i in range(0,100):
    print(i)
    env.reset()
    playerGo.reset()
    playerGo2.reset() 
    done = False
    while not done:
        action = playerGo.take_turn_2()
        _, _, done, _ = env.step(action)
        env.render('terminal')
        playerGo2.opponent_turn_update(action)
                
        if done:
            break
        
        action = playerGo2.take_turn_2()
        _, _, done, _ = env.step(action)
        env.render('terminal')
        playerGo.opponent_turn_update(action)

    playerGo.move_count = 0
    playerGo2.move_count = 0
    env.winner()
    if env.winner() == 1:
        playerGo_win += 1
    elif env.winner() == -1:
        playerGo2_win += 1
    else:
        draws +=1


# Plot the results in a bar graph
objects = ('Go (black)', 'Go2 (white)', 'Draws')
y_pos = np.arange(len(objects))
performance = [playerGo_win, playerGo2_win, draws]
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Wins')
plt.title('Wins with Go vs Go2')
plt.show()

