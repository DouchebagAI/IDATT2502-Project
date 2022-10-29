import json
import warnings
warnings.filterwarnings("ignore")
import gym
from models.MCTS import *
from models.game_manager import *
go_env = gym.make('gym_go:go-v0', size=5, komi=0, reward_method='real')

black = MCTS(go_env,"black", Type.BLACK)
white = MCTS(go_env,"white", Type.WHITE)
gm = GameManager(go_env, black, white)
gm.train(x=10)
print("White tree")
gm.white.R.check_ns()
print("Black tree")
gm.black.R.check_ns()
print("White tree")
#gm.white.R.print_node(0)
print("Black tree")
#gm.black.R.print_node(0)
#black.monte_carlo_tree_search(10000)
#gm.play_as_white()

