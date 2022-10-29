from models.node import Node, Type
from operator import attrgetter
import numpy as np
from enum import Enum



# Different stages for what step to do

class Stage(Enum):
    UNDEFINED = 0
    TRAVERSE = 1
    ROLLOUT = 2


class MCTS:
    # Inits a MCTS
    def __init__(self, env, name,type):
        self.stage = Stage.UNDEFINED
        self.env = env
        self.R = Node(None, None)
        self.currentNode = self.R

        self.type = type
        self.name = name

    # If there are children, choose the best child (move/action)
    # If not, create a new child with a random action
    def traverse_step(self, node: Node, render=False):
        if len(node.children) != 0:
            new_node = node.best_child(self.type)
            if new_node.get_value(self.type) > 1:
                return new_node

        new_node = Node(node, self.rollout_policy())
        node.children.update({(new_node.action, new_node)})
        return new_node

    # No more children, creating a random action/node
    def rollout_step(self, node):
        action = self.rollout_policy()
        new_node = Node(node, action)
        node.children.update({(action, new_node)})
        return new_node

    def rollout_policy(self):
        return self.env.uniform_random_action()

    # Checks if game is ended
    def is_terminal(self):
        return self.env.game_ended()

    # Updates the node values
    def backpropagate(self, node, v):
        if node.is_root():
            node.n += 1
            self.stage = Stage.TRAVERSE
            self.currentNode = node
        else:
            node.update_node(v)
            self.backpropagate(node.parent, v)

    # Pics or creates a new child node based on opponents turn
    def opponent_turn_update(self, action):
        if action in self.currentNode.children.keys():
            self.currentNode = self.currentNode.children[action]
        else:
            new_node = Node(self.currentNode, action)
            self.currentNode.children.update({(action, new_node)})
            self.currentNode = new_node

    # Takes a turn, either by traversing or rollouting
    # Returns action
    def take_turn(self, render=False):
        if self.stage == Stage.UNDEFINED:
            self.stage = Stage.TRAVERSE
            self.currentNode = self.traverse_step(self.currentNode, render)
        elif self.stage == Stage.TRAVERSE:
            # Changes state from traverse to rollout if no more children
            if len(self.currentNode.children) == 0:
                self.currentNode = self.rollout_step(self.currentNode)
                self.stage = Stage.ROLLOUT
            else:
                self.currentNode = self.traverse_step(self.currentNode, render)
        elif self.stage == Stage.ROLLOUT:
            self.currentNode = self.rollout_step(self.currentNode)

        return self.currentNode.action
