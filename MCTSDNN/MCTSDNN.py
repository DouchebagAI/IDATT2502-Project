from MCTSDNN.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np

class GoNN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.size = size
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.Conv2d(size**2+1, size**3+size, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7*7, 1024),
            nn.Flatten(),
            nn.Linear(1*1024, 10)
        )
        self.dl1 = nn.Linear(size**2, size**3)
        self.dl2 = nn.Linear(size**3, size**3)
        self.output_layer = nn.Linear(size**3, size**2+1)

    def logits(self, x):
        x = self.dl1(x)
        x = torch.relu(x)
        x = self.dl2(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

class MCTSDNN:

    def __init__(self, env):
        self.size = 3
        self.model = GoNN(self.size)
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

    def take_turn_play(self):
        action : int
        if len(self.current_node.children) == 0:
            action = self.play_policy(self.current_node)
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
        
        index = 0
        while(index < len(self.current_node.children)):
            action = list(self.current_node.children.keys())[index]
            node = self.current_node.children[action]
            simulated_node = self.train_simulate(node)
            self.current_node.children.update({(simulated_node.action, simulated_node)})
            index += 1
            
        self.train_model(self.current_node)
        self.current_node = self.current_node.best_child(self.get_type())
        return self.current_node.action
           
    def train(self, n = 10):
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
            self.reset()
    
    def play_policy(self, node: Node):
        #env_copy = copy.deepcopy(self.env)
        #state, _, _, _ = env_copy.step(node.action)
        node.state = self.env.state()
        x_tens = torch.tensor(node.state[0], dtype=torch.float)
        y = self.model.f(x_tens.reshape(-1,self.size**2))
        if self.get_type() == Type.BLACK:
            index = np.argmax(y.detach())
        else:
            index = np.argmin(y.detach())
        print(index.item())
        #value head
            # input state- lag - splitter i to 
                # ploicy head -> actions
                # value head -> verdi (om bra state)
                    # returnerer istedenfor å rolloute ned til terminal state
            # state probability pairs
            # rolling buffer
            # Trene på batch
            # legge i testsdata
            # sammenligne forskjellige mpter å velge på (prob, argmax)


        return index.item() 

    
    def train_simulate(self, node):
        
        env_copy  = copy.deepcopy(self.env)
        node.parent.state = env_copy.state()
        state, _, done, _ = env_copy.step(node.action)
    
        node.state = state
        
        while not done:
            state, _, done, _= env_copy.step(env_copy.uniform_random_action())
        self.backpropagate(node, env_copy.winner())
        return node
    
    
    def train_model(self, node: Node):
       
        y_train = np.zeros((self.size**2)+1)
        for n in node.children.values():
            y_train[n.action] = n.get_value_default(self.get_type())

        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(self.model.parameters(), 0.001)
        for epoch in range(5):
            x_tens = torch.tensor(node.state, dtype=torch.float)
            
            y_tens = torch.tensor(y_train, dtype=torch.float)
           
            self.model.loss(x_tens[0].reshape(-1,self.size**2),y_tens.reshape(-1, self.size**2+1)).backward()  # Compute loss gradients
            optimizer.step()  # Perform optimization by adjusting W and b,
            optimizer.zero_grad()  # Clear gradients for next step

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



    