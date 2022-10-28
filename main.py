import gym
from models.MCTS import *
from models.game_manager import *
go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')

black = MCTS(go_env)
white = MCTS(go_env)
gm = GameManager(go_env, black, white)
gm.train(x=1000)
gm.play_as_white()

