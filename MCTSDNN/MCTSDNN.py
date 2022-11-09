from MCTSDNN.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np
import gym


class GoNN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.size = size
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.Linear(9, 36),
            nn.Linear(36, 36),
            nn.Linear(36, 10)
            #nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Conv2d(32, 64, kernel_size=5, padding=2),
            #nn.ReLU(),
            #nn.Dropout(0.2),
            #nn.Conv2d(64, 128, kernel_size=5, padding=2),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2),
            #nn.Flatten(),
            #nn.Linear(64, 1024),
            #nn.Flatten(),
            #nn.Linear(1*1024, 10)
        )

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

class MCTSDNN:

    def __init__(self, env : gym.Env):
        self.size = 3
        self.model = GoNN(self.size)
        self.move_count = 0
        self.env = env
        self.R = Node(None, None)
        self.current_node = self.R
        self.node_count = 0
        self.states = []


    def take_turn(self):
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
            action = self.play_policy_greedy(self.current_node)
            new_node = Node(self.current_node, action)
            self.current_node.children.update({(action, new_node)})
            self.current_node = new_node
        else:
            self.current_node = self.current_node.best_child(self.get_type())
            action =  self.current_node.action
            valid_moves = self.env.valid_moves()
        self.move_count += 1
        return action

    def expand(self):
        valid_moves = self.env.valid_moves()
        # Create all valid children nodes
        
        for move in range(len(valid_moves)):
             if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1
        
        index = 0
        while(index < len(self.current_node.children)):
            action = list(self.current_node.children.keys())[index]
            node = self.current_node.children[action]
            simulated_node = self.simulate(node)
            self.current_node.children.update({(simulated_node.action, simulated_node)})
            index += 1
            
        self.current_node = self.current_node.best_child(self.get_type())
        
        """
        for i in range(100):
            #Traversere til leaf-node
            print("Sim round: ", i)
            while(len(self.current_node.children) > 0):
                self.current_node = self.current_node.best_child(self.get_type())
            
            #Velg beste action fra NN
            action = self.play_policy_greedy(self.current_node)
            #Expand
            new_node = Node(self.current_node, action)
            
            self.current_node.children.update({(new_node.action, new_node)})
            #Simulere
            self.simulate(new_node) 
            self.current_node = self.R
                
        while(len(self.current_node.children) > 0):
                self.current_node = self.current_node.best_child(self.get_type())
        """
        
        
        return self.current_node.action
           
   
            
    
    def play_policy_greedy(self, node: Node):
        
        #env_copy = copy.deepcopy(self.env)
        #state, _, _, _ = env_copy.step(node.action)
        node.state = self.env.state()[0] - self.env.state()[1]
        x_tens = torch.tensor(node.state, dtype=torch.float)
        y = self.model.f(x_tens.reshape(-1,9).float())
        if self.get_type() == Type.BLACK:
            index = np.argmax(y.detach())
        else:
            index = np.argmin(y.detach())
        
        valid_moves = self.env.valid_moves()
        
        if valid_moves[index.item()] == 0:
            return self.env.uniform_random_action()
        #value head
            # input state- lag - splitter i to 
                # policy head -> actions
                # value head -> verdi (om bra state)
                    # returnerer istedenfor å rolloute ned til terminal state
            # state probability pairs
            # rolling buffer
            # Trene på batch
            # legge i testsdata
            # sammenligne forskjellige mpter å velge på (prob, argmax)
        return index.item() 
    
    def simulate(self, node):
        
        env_copy  = copy.deepcopy(self.env)
        #node.parent.state = env_copy.state()[0] - env_copy.state()[1]
        #self.states.update({(str(env_copy.state()), node.parent)})
        
        state, _, done, _ = env_copy.step(node.action)
    
        node.state = state[0] - state[1]
        self.states.append((state, node))
        
        while not done:
            state, _, done, _= env_copy.step(env_copy.uniform_random_action())
            
        self.backpropagate(node, env_copy.winner())
        return node

    def get_training_data(self):
        x_train = []
        
        y_train = np.zeros([len(self.states), self.size**2+1])
        for i in range(len(self.states)):
            for n in self.states[i][1].children.values():
                y_train[i][n.action] = n.get_value_default(self.get_type())
            x_train.append(list(self.states)[i][1].state)

        x_tens = torch.tensor(x_train, dtype=torch.float).reshape(-1, 9)
        
        y_tens = torch.tensor(y_train, dtype=torch.float)
    
        batch = 100
        x_train_batches = torch.split(x_tens, batch)
        y_train_batches = torch.split(y_tens, batch)
        return x_train_batches, y_train_batches
        
    def train_model(self):
        #batch = 100
        
        x_train_batches, y_train_batches = self.get_training_data()
        
        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(self.model.parameters(), 0.001)
        
        for _ in range(40):
            for batch in range(len(x_train_batches)):
                self.model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
                optimizer.step()  # Perform optimization by adjusting W and b,
                optimizer.zero_grad()  # Clear gradients for next step
            print(f"Loss: {self.model.loss(x_train_batches[batch], y_train_batches[batch])}")
            

        
    def backpropagate(self, node: Node, v):
        
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        
        node.n += 1
        return node 
    
    def get_type(self):
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE
    
    def train(self, n=10):
        for i in range(n):
            print(f"Training round: {i}")
            # Nullstiller brettet
            self.env.reset()
            done = False
            while not done:              
                # Gjør et trekk
                action = self.take_turn()
                _, _, done, _ = self.env.step(action)

            self.backpropagate(self.current_node, self.env.winner())
            self.train_model()
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



    