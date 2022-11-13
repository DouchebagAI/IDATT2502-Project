import math
import random

from MCTSDNN.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np
import gym
#0.76
class GoDCNN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.size = size
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.Conv2d(6, size**2, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(size**2, size**3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(size ** 3, size ** 4, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(625, size**4),
            nn.Flatten(),
            nn.Linear(1*size**4, size**2+1)
        )
        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x),  y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

#0.665
class GoCNN(nn.Module):
    def __init__(self, size=3):
        super().__init__()
        self.size = size
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.Conv2d(6, size ** 2, kernel_size=5, padding=2),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(size ** 2, size ** 3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(size ** 3, size ** 4, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(size ** 4, size ** 5, kernel_size=5, padding=2),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(3125, size ** 4),
            nn.Flatten(),
            nn.Linear(1 * size ** 4, size ** 2 + 1)
        )
        """
        self.logits = nn.Sequential(
            nn.Conv2d(6, size**2, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            #nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(size**2, size**3, kernel_size=5, padding=2),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(125, size**4),
            nn.Flatten(),
            nn.Linear(1*size**4, size**2+1)
        )
        """
        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x),  y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())
#0.79
class GoNN(nn.Module):
    def __init__(self, size=3, kernel_size = 3):
        super().__init__()
        self.size = size
        lin = 100 if kernel_size == 5 else 225
        # Conv
        # Relu
        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(6, size**2, kernel_size=kernel_size, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(lin, size**4),
            nn.Linear(1*size**4, size**2+1)
        )
        self.logits.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

class MCTSDNN:

    def __init__(self, env : gym.Env, size, model, kernel_size = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        if model is "Go":
            self.model = GoNN(self.size, kernel_size=kernel_size).to(self.device)
        if model is "Go2":
            self.model = GoCNN(self.size).to(self.device)
        if model is "Go3":
            self.model = GoDCNN(self.size).to(self.device)

        self.move_count = 0
        self.env = env
        self.R = Node(None, None)
        self.current_node = self.R
        self.node_count = 0
        self.states = []
        self.losses = []
        self.accuracy = []


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
            action = self.play_policy_greedy(self.env)
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
        
        # Create all valid children nodes
        valid_moves = self.env.valid_moves()
        for move in range(len(valid_moves)):
             if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1

        for i in range(100):
            self.simulate(self.current_node)

        self.current_node.state = self.env.state()
        self.states.append((self.env.state(), self.current_node))



        for i in self.current_node.children.values():
            env_copy  = copy.deepcopy(self.env)
            state, reward, done, _ = env_copy.step(i.action)
            i.state = state
            self.states.append((state, i))


        self.current_node = self.current_node.best_child(self.get_type())
        
        """

        #valid_moves = self.env.valid_moves()
        #Velg beste action fra NN
        action = self.play_policy_greedy(self.env)
        #Expand
        new_node = Node(self.current_node, action)
        self.current_node.children.update({(new_node.action, new_node)})
        for i in range(100):
            #Simulere 100 ganger
            #print("Sim round: ", i)
            self.simulate(new_node)
            
        
        self.current_node = new_node
        
        """
        return self.current_node.action

    def simulate(self, node):


        env_copy = copy.deepcopy(self.env)

        actionFromNode = node.best_child(self.get_type()).action
        state, reward, done, _ = env_copy.step(actionFromNode)

        while not done:
            action = env_copy.uniform_random_action()
            state, _, done, _ = env_copy.step(action)

        self.backpropagate(node.children[actionFromNode], env_copy.winner())
        return node
            
    
    def play_policy_greedy(self, env):
        
        #env_copy = copy.deepcopy(self.env)
        #state, _, _, _ = env_copy.step(node.action)
        #node.state = self.env.state()[0] - self.env.state()[1
        x_tens = torch.tensor(env.state(), dtype=torch.float).to(self.device)

        y = self.model.f(x_tens.reshape(-1, 6,self.size, self.size).float())
        if self.get_type() == Type.BLACK:
            index = np.argmax(y.cpu().detach())
        else:
            index = np.argmin(y.cpu().detach())
        
        valid_moves = env.valid_moves()
        
        if valid_moves[index.item()] == 0.0:
            #print("Invalid move")
            return env.uniform_random_action()
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
        if index.item() == 9:
            #print("Passed")
            pass
        return index.item() 
    


    def get_training_data(self):
        x_train = []
        random.shuffle(self.states)
        y_train = []

        #print(self.states[0][1])
        t = 0
        for i in range(len(self.states)):
            y_t = np.zeros(self.size**2+1)
            for n in self.states[i][1].children.values():
                y_t[n.action] = n.get_value_default(n.get_type())
                #print(y_t)

            if sum(y_t) != 0:
                y_train.append(y_t)
                x_train.append(list(self.states)[i][1].state)
                #print(f"board: {x_train[t][0]-x_train[t][1]}x: {x_train[t][2]}, y: {y_train[t]}")
                t += 1




        
        limit = math.floor(len(x_train)/3)
        print(len(x_train))
        x_tens = torch.tensor(x_train, dtype=torch.float).reshape(-1,6,self.size, self.size).float().to(self.device)

        y_tens = torch.tensor(y_train, dtype=torch.float).float().to(self.device)


        x_test = x_tens[limit*2:]
        y_test = y_tens[limit*2:]

        batch = 200
        x_train_batches = torch.split(x_tens[:limit*2], batch)
        #print(len(x_train_batches))
        #print(x_train_batches[0])
        y_train_batches = torch.split(y_tens[:limit*2], batch)
        return x_train_batches, y_train_batches, x_test, y_test
        
    def train_model(self):

        x_train_batches, y_train_batches, x_test, y_test = self.get_training_data()

        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(self.model.parameters(), 0.0001)
        
        for _ in range(1000):
            for batch in range(len(x_train_batches)):
                self.model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
                self.losses.append(self.model.loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                optimizer.step()  # Perform optimization by adjusting W and b,
                optimizer.zero_grad()  # Clear gradients for next step
            self.accuracy.append(self.model.accuracy(x_test, y_test).cpu().detach())
        #print(f"Loss: {self.model.loss(x_test, y_test)}")
        print(f"Accuracy: {self.model.accuracy(x_test, y_test)}")
            
    def get_accuracy(self):
        if len(self.accuracy) == 0:
            return 0
        return np.max(self.accuracy)
        
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
            if i % 5 is 0 and i != 0:
                #self.train_model()
                pass
        
        self.train_model()
        print(len(self.states))
    
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



    