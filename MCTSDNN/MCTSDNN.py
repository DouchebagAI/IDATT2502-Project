import math
import random
from models.GoCNN import GoCNN
from models.GoDCNN import GoDCNN
from models.GoNN import GoNN
from models.CNNValue import GoCNNValue
from MCTSDNN.Node import Node, Type
from enum import Enum
import torch
import torch.nn as nn
import copy
import numpy as np
import gym
"""
1. Reskaler modeller -> mindre ✅
2. Lagre treningsdata som primitive dataverdier - Viktig❕
3. Slett treet etter 1 gjennomkjøring - 
4. Bruk value head-modellen til å predikere win/loss 
5. Undersøk batch size
6. Refaktorere kode (bruker mye av samme metoder 2 ganger)
7. Lagre treningsdata / testdata som json
8. Kernel-size -> 3
9. Bytt til value head model etter x-antall gjennomkjølringer
"""

class MCTSDNN:

    def __init__(self, env : gym.Env, size, model, kernel_size = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        if model is "Go":
            self.model = GoNN(self.size, kernel_size=kernel_size).to(self.device)
        if model is "Go2":
            self.model = GoCNN(self.size).to(self.device)

        self.value_model = GoCNNValue(self.size, 5).to(self.device)
        self.move_count = 0
        self.env = env
        self.R = Node(None, None)
        self.current_node = self.R
        self.node_count = 0
        self.amountOfSims = 0

        self.training_win = []
        self.test_win = []
        
        self.model_losses = []
        self.model_accuracy = []
        
        self.value_model_losses = []
        self.value_model_accuracy = []
        
        self.training_data = []
        self.test_data = []


    def take_turn(self):
        # Hvis ingen barn, exlandasd
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
        
        for i in range(50):
            self.simulate(self.current_node)
            
        self.amountOfSims += 1
        
        # Add to current node
        if(random.randint(1,99) <= 33):
            y_t = np.zeros(1)
            y_t[0] = self.current_node.v
            self.test_data.append((self.env.state(), self.get_target(self.current_node)))
            self.test_win.append((self.env.state(), y_t))
        else:
            y_t = np.zeros(1)
            y_t[0] = self.current_node.v
            self.training_data.append((self.env.state(), self.get_target(self.current_node)))
            self.training_win.append((self.env.state(), y_t))

        # Add to children to current node
        for i in self.current_node.children.values():
            env_copy  = copy.deepcopy(self.env)
            state, reward, done, _ = env_copy.step(i.action)
                                
            if(random.randint(1,99) <= 33):
                y_t = np.zeros(1)
                y_t[0] = i.v
                self.test_data.append((state, self.get_target(i)))
                self.test_win.append((state, y_t))
            else:
                y_t = np.zeros(1)
                y_t[0] = i.v
                self.training_data.append((state, self.get_target(i)))
                self.training_win.append((state,y_t))
                
                
    

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


    def get_target(self, node: Node):
        y_t = np.zeros(self.size**2+1)
        for i in node.children.values():
            y_t[i.action] = i.get_value_default(self.get_type())
        
        return y_t
        

    def simulate(self, node):
        print("Simulating", self.amountOfSims)
        env_copy = copy.deepcopy(self.env)
        actionFromNode = node.best_child(self.get_type()).action
        state, reward, done, _ = env_copy.step(actionFromNode)
        
        if( self.amountOfSims > 30):
            x_tens = torch.tensor(state, dtype=torch.float).to(self.device)
            v = self.value_model.f(x_tens.reshape(-1, 6,self.size, self.size).float()).cpu().detach().item()
            print(v)
            self.backpropagate(node.children[actionFromNode], v)
        else:
            while not done:
                action = self.play_policy_greedy(env_copy)
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
            index = np.argmax(y.cpu().detach()*env.valid_moves())
        else:
            index = np.argmin(y.cpu().detach()*env.valid_moves())
        
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
    
        return index.item() 
    
    def value_head(self, state):
        pass

    def value_data_to_tensor(self, data):
        x_train = []
        random.shuffle(data)
        y_train = []

        #print(self.states[0][1])
        t = 0
        for i in range(len(data)):
            y_t = data[i][1]
            y_train.append(y_t)
            x_train.append(list(data)[i][0])

        x_tens = torch.tensor(x_train, dtype=torch.float).reshape(-1,6,self.size, self.size).float().to(self.device)

        y_tens = torch.tensor(y_train, dtype=torch.float).float().to(self.device)

        return x_tens, y_tens

    def data_to_tensor(self, data):
        x_train = []
        random.shuffle(data)
        y_train = []

        #print(self.states[0][1])
        for i, tup in enumerate(data):
            
            y_train.append(tup[1])
            x_train.append(tup[0])

        x_tens = torch.tensor(x_train, dtype=torch.float).reshape(-1,6,self.size, self.size).float().to(self.device)
    
        y_tens = torch.tensor(y_train, dtype=torch.float).float().to(self.device)

        return x_tens, y_tens

    def get_training_data(self, train, test):
        x_train, y_train = self.data_to_tensor(train)
        x_test, y_test = self.data_to_tensor(test)

        batch = 32
        x_train_batches = torch.split(x_train, batch)
        #print(len(x_train_batches))
        #print(x_train_batches[0])
        y_train_batches = torch.split(y_train, batch)
        return x_train_batches, y_train_batches, x_test, y_test


    def train_model(self, model, function, loss_list, acc_list):
        x_train_batches, y_train_batches, x_test, y_test = function
        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)
        
        for _ in range(1000):
            for batch in range(len(x_train_batches)):
                model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
                loss_list.append(model.loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                optimizer.step()  # Perform optimization by adjusting W and b,
                optimizer.zero_grad()  # Clear gradients for next step
            acc_list.append(model.accuracy(x_test, y_test).cpu().detach())
        # print(f"Loss: {self.model.loss(x_test, y_test)}")
        print(f"Accuracy: {model.accuracy(x_test, y_test)}")
        print(len(x_test))
        

    def get_accuracy(self):
        if len(self.model_accuracy) == 0:
            return 0
        return np.max(self.model_accuracy)
        
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
            print("Training network")
            self.train_model(self.model, self.get_training_data(self.training_data, self.test_data), self.model_losses, self.model_accuracy)
            self.train_model(self.value_model, self.get_training_data(self.training_win, self.test_win), self.value_model_losses, self.value_model_accuracy)
            self.R = Node(None, None)
            self.reset()
           
        print(len(self.training_win))
    
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