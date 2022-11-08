from MCTS.Node import Node, Type
from enum import Enum
import copy


import time

class MCTS:
    
    def __init__(self, env):
        self.move_count = 0
        self.env = env
        self.R = Node(None, None)
        self.current_node = self.R
        self.node_count = 0

    def take_turn(self):
        action : int
        if len(self.current_node.children) == 0:
            action = self.expand()
        else:
            self.current_node = self.current_node.best_child(self.get_type())
            action =  self.current_node.action
        self.move_count += 1
        return action

    def expand(self):
        valid_moves = self.env.valid_moves()
        for move in range(len(valid_moves)):
             if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1
        
        start = time.time()
        index = 0
        while(index < len(self.current_node.children)):
            action = list(self.current_node.children.keys())[index]
            node = self.current_node.children[action]
            simulated_node = self.simulate(node)
            self.current_node.children.update({(simulated_node.action, simulated_node)})
            index += 1
            
        self.current_node = self.current_node.best_child(self.get_type())
        return self.current_node.action
           
    def simulate(self, node):
        env_copy = copy.deepcopy(self.env)
        _, _, done, _ = env_copy.step(node.action)
        while not done:
            _, _, done, _ = env_copy.step(env_copy.uniform_random_action())
        self.backpropagate(node, env_copy.winner())
        return node
            
    def backpropagate(self, node: Node, v):
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        node.n += 1
        return node 
    
    def get_type(self):
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE
    
    def opponent_turn_update(self, move):
        if move in self.current_node.children.keys():
            self.current_node = self.current_node.children[move]
        else:
            new_node = Node(self.current_node, move)
            self.current_node.children.update({(move, new_node)})
            self.current_node = new_node
        self.moveCount += 1
        
        
    def reset(self):
        self.moveCount = 0
        self.current_node = self.R
        
    # Metode for å visualisere treet
    def print_tree(self):
        self.R.print_tree()