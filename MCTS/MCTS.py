import math
import random
from models import GoCNN, GoDCNN, GoNN
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
        #print(f"move count {self.move_count} type: {self.get_type()}")

        # Hvis ingen barn, exland
        # Hvis ikke, velg det barnet med høyest verdi
        action : int
        if len(self.current_node.children) == 0:
            action = self.expand()
        else:
            self.current_node = self.current_node.best_child(self.get_type())
            action =  self.current_node.action
        self.move_count += 1

        return action

    def take_turn_play(self):
        # print(f"move count {self.move_count} type: {self.get_type()}")
        # Hvis ingen barn, velg en greedy policy
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
        
        # Create all valid children nodes
        valid_moves = self.env.valid_moves()
        for move in range(len(valid_moves)):
             if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1

        for _ in range(300):
            self.simulate(self.current_node)
        

        self.current_node = self.current_node.best_child(self.get_type())
        
        return self.current_node.action

    def simulate(self, node):
        env_copy = copy.deepcopy(self.env)

        actionFromNode = node.best_child(self.get_type()).action
        _, _, done, _ = env_copy.step(actionFromNode)

        while not done:
            action = env_copy.uniform_random_action()
            _, _, done, _ = env_copy.step(action)

        self.backpropagate(node.children[actionFromNode], env_copy.winner())
        return node
            
    
    def play_policy_greedy(self, env):
        return env.uniform_random_action()

    def backpropagate(self, node: Node, v):
        
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        
        node.n += 1
        return node 
    
    def get_type(self):
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE
    
    def train(self, n):
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
        self.move_count += 1
        if move in self.current_node.children.keys():
            self.current_node = self.current_node.children[move]
        else:
            new_node = Node(self.current_node, move)
            self.current_node.children.update({(move, new_node)})
            self.current_node = new_node

    def reset(self):
        self.move_count = 0
        self.current_node = self.R
        
    # Metode for å visualisere treet
    def print_tree(self):
        self.R.print_tree()