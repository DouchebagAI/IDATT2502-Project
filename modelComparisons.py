import gym
import numpy as np
from MCTS.MCTS import MCTS as MCTSTREE
import matplotlib.pyplot as plt
from MCTSDNN.MCTSDNN import MCTSDNN
from GameManager import GameManager
size = 5
go_env = gym.make('gym_go:go-v0', size=size, komi=0, reward_method='real')

gm = GameManager(go_env)


# med og uten tre mot hverandre





# ulike policyer mot hverandre





# sammenlign rent tre mot tre med nettverj





# sammenlign rent tre mot tre med nettverk og tre





# Ulik loss funksjon mot hverandre