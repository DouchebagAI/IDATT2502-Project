import json

import gym
from models.MCTS import *
from models.game_manager import *
go_env = gym.make('gym_go:go-v0', size=5, komi=0, reward_method='real')

black = MCTS(go_env, y=1)
white = MCTS(go_env, y=-1)
gm = GameManager(go_env, black, white)
gm.train(x=10000)
#black.monte_carlo_tree_search(10000)
gm.play_as_white()

