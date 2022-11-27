import gym
import numpy as np
from MCTS.MCTS import MCTS
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager

size = 5

env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(env)

def plot_training(mcts: MCTSDNN, title, loss, acc):
    losses = np.array(loss)
    accuracy = np.array(acc)
    plt.figure()
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

no_data_aug = MCTSDNN(env, size, "Go2",data_augment=False)
data_aug = MCTSDNN(env, size, "Go2",data_augment=True)

no_data_aug.train(7)
data_aug.train(7)
networkWins = 0
ties = 0
treeWins = 0
for i in range(0,100):
    env.reset()
    print(i)
    data_aug.reset()
    no_data_aug.reset() 
    done = False
    while not done:
        action = no_data_aug.play_policy_prob(env)
        _, _, done, _ = env.step(action)
        
        data_aug.opponent_turn_update(action)
                
        if done:
            break
        
        action = data_aug.play_policy_prob(env)
        _, _, done, _ = env.step(action)
        no_data_aug.opponent_turn_update(action)
    
    
    if env.winner() == -1:
        networkWins += 1
    elif env.winner() == 0:
        ties += 1
    else:
        treeWins += 1


print("DataAugmentation wins: " + str(networkWins) + " out of 100 games")

plot_training(no_data_aug, "No Data Augmentation", no_data_aug.model_losses, no_data_aug.model_accuracy)

plot_training(data_aug, "W/ Some Data Augmentation (Only flip)", data_aug.model_losses, data_aug.model_accuracy)