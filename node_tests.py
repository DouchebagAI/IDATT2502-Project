from models.MCTS import MCTS, Type
from modelsV2.node import Node

mcts = MCTS(None, "black", Type.BLACK)

def init_nodes():
    mcts.R.n = 6
    mcts.R.children.update({(1, Node(mcts.R, 1)),
                            (2, Node(mcts.R, 2)),
                            (3, Node(mcts.R, 3)),
                            (4, Node(mcts.R, 4)),
                            (5, Node(mcts.R, 5))})
    mcts.R.children[1].update_node(1)
    mcts.R.children[1].update_node(0)
    assert mcts.R.children[1].n == 2
    mcts.R.children[2].update_node(1)
    mcts.R.children[3].update_node(-1)
    mcts.R.children[3].update_node(-1)
    assert mcts.R.children[3].n == 2
    mcts.R.children[4].update_node(0)
    mcts.R.children[5].update_node(0)


def test_best_child(y = Type.BLACK):
    init_nodes()
    print(mcts.R.best_child(y))
    print()
    print(mcts.R.children[1].get_value(y))
    print(mcts.R.children[2].get_value(y))
    print(mcts.R.children[3].get_value(y))
    print(mcts.R.children[4].get_value(y))
    print(mcts.R.children[5].get_value(y))
    print()

test_best_child(Type.BLACK)
test_best_child(Type.WHITE)