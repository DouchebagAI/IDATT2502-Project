from models.node import Node
from operator import attrgetter
import numpy as np


class MCTS:

    def __init__(self, env):
        self.env = env
        self.R = Node(None, None)
        self.currentNode = self.R

    def monte_carlo_tree_search(self, x=1000, render=False):
        for i in range(x):
            leaf = self.traverse(self.R, render)
            simulation_result = self.rollout(leaf)
            self.backpropagate(leaf, simulation_result)
        return self.R.best_child()

    def traverse(self, node: Node, render=False):
        done = False
        while len(node.children) != 0 and not done:
            node = node.best_child()
            state, reward, done, info = self.env.step(node.action)
            if done:
                print("Hello")
            if render:
                self.env.render('terminal')

        new_node = Node(node, self.rollout_policy())
        node.children.append(new_node)
        return new_node

    def rollout(self, node):
        while not self.is_terminal():
            action = self.rollout_policy()
            new_node = Node(node, action)
            node.children.append(new_node)
            node = new_node
            self.env.step(node.action)
        return self.env.reward()

    def rollout_policy(self):
        return self.env.uniform_random_action()

    def is_terminal(self):
        return self.env.game_ended()

    def backpropagate(self, node, v):
        if node.is_root(): return
        node.stats = node.update_node(v)
        self.backpropagate(node.parent, v)
