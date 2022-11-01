from models.MCTS import MCTS
from modelsV2.node import Node

mcts_black = MCTS(None, "black")
mcts_white = MCTS(None, "white", -1)


def init():
    n1 =Node(mcts_black.R, 1)
    mcts_black.R.children.update({(1, n1)})
    n2 = Node(n1, 2)
    n1.children.update({(2, n2)})
    n3 = Node(n2, 3)
    n2.children.update({(3, n3)})
    return n3

def backprop_test():
    leaf = init()
    mcts_black.backpropagate(leaf, 1)
    mcts_black.backpropagate(leaf, 1)
    mcts_black.backpropagate(leaf, -1)
    mcts_black.R.print_node(0)



backprop_test()
