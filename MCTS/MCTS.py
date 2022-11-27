import math
import random
from MCTS.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np
import gym


class MCTS:

    def __init__(self, env : gym.Env):
        self.move_count = 0
        self.env = env
        self.R = Node(None, None)
        self.current_node = self.R
        self.node_count = 0

    def take_turn(self):
        """
        Method to take a turn while training
        If no children expand, if children, pick best
        """
        action : int
        if len(self.current_node.children) == 0:
            action = self.expand()
        else:
            self.current_node = self.current_node.best_child(self.get_type())
            action =  self.current_node.action
        self.move_count += 1

        return action

    def take_turn_play(self):
        """
        Method to take a turn while paying (not training)
        If no children, choose random move, if not, choose best child
        """
        action : int
        if len(self.current_node.children) == 0:
            action = self.play_policy_greedy(self.env)
            new_node = Node(self.current_node, action)
            self.current_node.children.update({(action, new_node)})
            self.current_node = new_node
        else:
            self.current_node = self.current_node.best_child(self.get_type())
            action =  self.current_node.action
        self.move_count += 1
        return action

    def expand(self):
        """
        Method to expand Search Tree
        Creates all nodes for the new node
        Simulates 100 times
        :return: best child's action
        """
        # Create all valid children nodes
        valid_moves = self.env.valid_moves()
        for move in range(len(valid_moves)):
             if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1

        for _ in range(100):
            self.simulate(self.current_node)
        

        self.current_node = self.current_node.best_child(self.get_type())
        
        return self.current_node.action

    def simulate(self, node):
        """
        Method to simulate a game to terminal state from given node
        Backpropagates until root node when terminal state is reached

        :param node: The node to simulate from
        :return: node
        """
        env_copy = copy.deepcopy(self.env)

        actionFromNode = node.best_child(self.get_type()).action
        _, _, done, _ = env_copy.step(actionFromNode)

        while not done:
            action = env_copy.uniform_random_action()
            _, _, done, _ = env_copy.step(action)

        self.backpropagate(node.children[actionFromNode], env_copy.winner())
        return node
            
    
    def play_policy_greedy(self, env):
        """
        Play Policy
        Chooses random move
        """
        return env.uniform_random_action()

    def backpropagate(self, node: Node, v):
        """
        This is a function for backpropagating the MCTS.
        It will start in a leaf node and update the value based on the result of the game.
        It will continue to do it until it reaches the root node.

        :param node: The current node to be updated
        :param v: The result (win: 1, tie: 0, loss: -1)
        :return: node
        """
        
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        
        node.n += 1
        return node 
    
    def get_type(self):
        """
        Returns Black or White given which player turn it is
        """
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE
    
    def train(self, n):
        """
        Method to train model
        """
        for i in range(n):
            print(f"Training round: {i}")
            # Nullstiller brettet
            self.env.reset()
            done = False
            while not done:
                # Gjør et trekk
                action = self.take_turn()
                _, _, done, _ = self.env.step(action)
                    
                #self.env.render("terminal")
            self.backpropagate(self.current_node, self.env.winner())
            self.reset()
            
    
    
    def opponent_turn_update(self, move):
        """
        Method used when playing against another player to update turn counter and current node
        """
        self.move_count += 1
        if move in self.current_node.children.keys():
            self.current_node = self.current_node.children[move]
        else:
            new_node = Node(self.current_node, move)
            self.current_node.children.update({(move, new_node)})
            self.current_node = new_node

    def reset(self):
        """
        Resets move count and sets current node to Root
        """
        self.move_count = 0
        self.current_node = self.R
        
    def print_tree(self):
        self.R.print_tree()