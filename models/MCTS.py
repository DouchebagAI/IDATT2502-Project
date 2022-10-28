from models.node import Node
from operator import attrgetter
import numpy as np

from enum import Enum


class Stage(Enum):
    UNDEFINED = 0
    TRAVERSE = 1
    ROLLOUT = 2


class MCTS:

    def __init__(self, env):
        self.stage = Stage.UNDEFINED
        self.env = env
        self.R = Node(None, None)
        self.currentNode = self.R

    def monte_carlo_tree_search(self, x=100, render=False):
        for i in range(x):
            print(i)
            self.env.reset()
            leaf = self.traverse(self.currentNode, render)
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
                break
            if render:
                self.env.render('terminal')

        new_node = Node(node, self.rollout_policy())
        node.children.update({(new_node.action,new_node)})
        return new_node

    def traverse_step(self, node: Node, render=False):
        if len(node.children) != 0:
            node = node.best_child()
            state, reward, done, info = self.env.step(node.action)
            if render:
                self.env.render('terminal')
            return node

        new_node = Node(node, self.rollout_policy())
        node.children.update({(new_node.action,new_node)})
        return new_node

    def rollout(self, node):
        while not self.is_terminal():
            action = self.rollout_policy()
            new_node = Node(node, action)
            node.children.update({(new_node.action,new_node)})
            node = new_node
            self.env.step(node.action)
        return self.env.reward()

    def rollout_step(self, node):
        action = self.rollout_policy()
        new_node = Node(node, action)
        node.children.update({(new_node.action,new_node)})
        node = new_node
        self.env.step(node.action)

    def rollout_policy(self):
        return self.env.uniform_random_action()

    def is_terminal(self):
        return self.env.game_ended()

    def backpropagate(self, node, v):
        if node.is_root():
            self.currentNode = node
            return
        node.stats = node.update_node(v)
        self.backpropagate(node.parent, v)

    def opponent_turn_update(self, action):
        if action in self.currentNode.children.keys():
            self.currentNode = self.currentNode.children[action]
        else:
            new_node = Node(self.currentNode, self.rollout_policy())
            self.currentNode.children.update({(new_node.action,new_node)})
            self.currentNode = new_node

    def take_turn(self, render=False):
        if self.stage == Stage.UNDEFINED:
            self.stage = Stage.TRAVERSE
            self.currentNode = self.traverse_step(self.currentNode, render)
        if self.stage == Stage.TRAVERSE:
            if len(self.currentNode.children) == 0:
                self.rollout_step(self.currentNode)
                self.stage = Stage.ROLLOUT
            else:
                self.currentNode = self.traverse_step(self.currentNode, render)
        if self.stage == Stage.ROLLOUT:
            self.rollout_step(self.currentNode)
        return self.is_terminal()
