from GameManager import GameManager as gm
from MCTSDNN.MCTSDNN import MCTSDNN
import gym
import matplotlib as plt
import numpy as np

size = 5
number_of_players = 6

go_env = gym.make("gym_go:go-v0", size=size, komi=0, reward_method="real")
#playerDCNN = MCTSDNN(go_env, size, "Go2", kernel_size=3)
#playerDCNN.train(number_of_players)
gm = gm(go_env)
play_as_worst = gm.play_tournament(player=0, num_players=number_of_players)
play_as_best = gm.play_tournament(player=number_of_players-1, num_players=number_of_players)
# Plot the play_as_worst and play_as_best
plt.figure()
plt.title("Play tournament as worst and best model")
plt.xlabel("Model iteration")
plt.ylabel("Wins")
plt.plot(np.array(list(play_as_worst).keys()), np.array(list(play_as_worst).values()), label="Play as worst model", color="blue")
plt.plot(np.array(list(play_as_best).keys()), np.array(list(play_as_best).values()), label="Play as best model", color="green")
plt.legend()
plt.show()

