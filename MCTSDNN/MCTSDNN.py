import random
import uuid

from models.GoCNN import GoCNN
from models.GoNN import GoNN
from models.CNNValue import GoCNNValue
from MCTSDNN.Node import Node, Type
import torch
import torch.nn as nn
import copy
import numpy as np
import gym

"""
1. Reskaler modeller -> mindre ✅
2. Lagre treningsdata som primitive dataverdier - 
3. Slett treet etter 1 gjennomkjøring - ✅
4. Bruk value head-modellen til å predikere win/loss ✅
5. Undersøk batch size 🚼
6. Refaktorere kode (bruker mye av samme metoder 2 ganger)
7. Lagre treningsdata / testdata som json 
8. Kernel-size -> 3
9. Bytt til value head model etter x-antall gjennomkjølringer ✅
10. Mekke turnering mot seg selv, vise forbedring av nettverk
"""


class MCTSDNN:

    def __init__(self, env: gym.Env, size, model, kernel_size=3, prob_policy = True):
        self.prob_pol = prob_policy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model_name = model
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
        """
        This is a function for traversing the Monte Carlo Search Tree.
        If the current node had child nodes pick the child with the best value and return the given action.
        If the current node is a leaf node expand the tree and return the best action.
        :return: action
        """
        action: int
        parent_node = self.current_node
        for i in range(250):
            if len(self.current_node.children) == 0:
                # The current node has no child nodes
                action = self.expand()
                # expand
                # simulate
                # backpropagate
            else:
                # The current node has child nodes
                self.current_node = self.current_node.best_child(self.get_type())
                action = self.current_node.action
            
        self.move_count += 1
        return action

    def take_turn_play(self):
        """
        This is a function for choosing the best move with the trained neural network model.
        :return: action
        """
        # Hvis ingen barn, velg en greedy policy
        action : int
        if self.prob_pol:
            action = self.play_policy_prob(self.env)
        else:
            action = self.play_policy_greedy(self.env)

        self.move_count += 1
        return action

    def expand(self):
        """
        This is a function for expanding the MCTS.
        First it creates all valid child nodes, then it simulates games from the parent node
        in order to choose the best action.
        It will also create test and training data
        :return: The action og the best child node
        """
        # Create all valid children nodes
        valid_moves = self.env.valid_moves()
        for move in range(len(valid_moves)):
            if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1

        
        self.simulate(self.current_node)

        # Add to current node
        if (random.randint(1, 99) <= 20):
            y_t = np.zeros(1)
            y_t[0] = self.current_node.V()
            y = self.get_target(self.current_node)
            if np.sum(y) < 1000:
                self.test_data.append((self.env.state(), y))
            self.test_win.append((self.env.state(), y_t))
        else:
            y_t = np.zeros(1)
            y_t[0] = self.current_node.V()
            y = self.get_target(self.current_node)
            if np.sum(y) < 1000:
                self.training_data.append((self.env.state(), y))
            self.training_win.append((self.env.state(), y_t))

        # Add to children to current node
        for i in self.current_node.children.values():
            env_copy = copy.deepcopy(self.env)
            state, _, _, _ = env_copy.step(i.action)
            y_t = np.zeros(1)
            if not i.n == 0:
                y_t[0] = i.V()
            if (random.randint(1, 99) <= 20):
                y = self.get_target(i)
                if np.sum(y) != 0 and np.sum(y) < 1000:
                    self.test_data.append((state, y))
                if i.n > 5:
                    self.test_win.append((state, y_t))
            else:
                y = self.get_target(i)
                if np.sum(y) != 0 and np.sum(y) < 1000:
                    self.training_data.append((state, y))
                if i.n > 10:
                    self.training_win.append((state, y_t))

        self.current_node = self.current_node.best_child(self.get_type())
        
        return self.current_node.best_child(self.get_type()).action

    def get_target(self, node: Node):
        y_t = np.zeros(self.size ** 2 + 1)
        for i in node.children.values():
            y_t[i.action] = i.get_value_default(self.get_type())

        return y_t

    def simulate(self, node):
        """
        This is a function for simulating games from a given state.
        At first it will use the policy network to determine the best move, and play to terminal state before
        it backpropagates the result.
        However when the number of simulation is above three it will use the value network
        to predict the winner of the game, before it backpropogate correctly with the returned value of the value network.

        :param node: The parent node
        :return: The parent node with updated values
        """
        env_copy = copy.deepcopy(self.env)
        actionFromNode = node.best_child(self.get_type()).action
        state, _, done, _ = env_copy.step(actionFromNode)

        if self.amountOfSims > 9:
            x_tens = torch.tensor(state, dtype=torch.float).to(self.device)
            v = self.value_model.f(x_tens.reshape(-1, 6, self.size, self.size).float()).cpu().detach().item()
            if v > 0.5:
                self.backpropagate(node.children[actionFromNode], 1)
            if v < -0.5:
                self.backpropagate(node.children[actionFromNode], -1)
            else:
                self.backpropagate(node.children[actionFromNode], 0)
        else:
            i = 0
            while not done:
                if i > 100:
                    break
                if self.prob_pol:
                    action = self.play_policy_prob(env_copy)
                else:
                    action = self.play_policy_greedy(env.copy)
                state, _, done, _ = env_copy.step(action)
                i+= 1

            self.backpropagate(node.children[actionFromNode], env_copy.winner())
        return node

    def action_based_on_prob(self, actions_softmax):
        numb = np.random.rand()

        sum = 0
        for i, item in enumerate(actions_softmax):
            sum += item
            if numb < sum:
                return i
        return len(actions_softmax)

    def play_policy_prob(self, env):
        x_tens = torch.tensor(env.state(), dtype=torch.float).to(self.device)
        y = self.model.f_2(x_tens.reshape(-1, 6, self.size, self.size).float())


        #print(y.shape)
        y = y.cpu().detach()

        y = torch.multiply(y, torch.tensor(env.valid_moves()))
        #print(y)
        #print(env.valid_moves())
        #print(y[0])

        for i,u in enumerate(y[0]):
            if u == 0.0:
                y[0][i]= torch.finfo(torch.float64).min
        #print(y)

        sm = torch.softmax(torch.tensor(y, dtype=torch.float), dim= 1)
        action = self.action_based_on_prob(sm[0])

        return action

    def play_policy_greedy(self, env):

        # env_copy = copy.deepcopy(self.env)
        # state, _, _, _ = env_copy.step(node.action)
        # node.state = self.env.state()[0] - self.env.state()[1
        x_tens = torch.tensor(env.state(), dtype=torch.float).to(self.device)

        y = self.model.f(x_tens.reshape(-1, 6, self.size, self.size).float())
        index = np.argmax(np.multiply(y.cpu().detach(), env.valid_moves()))

        # value head
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

    def data_to_tensor(self, data):
        x_train = []
        random.shuffle(data)
        y_train = []

        # print(self.states[0][1])
        for i, tup in enumerate(data):
            x_train.append(tup[0])
            y_train.append(tup[1])

        x_tens = torch.tensor(x_train, dtype=torch.float).reshape(-1, 6, self.size, self.size).float().to(self.device)

        y_tens = torch.tensor(y_train, dtype=torch.float).float().to(self.device)

        return x_tens, y_tens

    def get_training_data(self, train, test):
        x_train, y_train = self.data_to_tensor(train)
        x_test, y_test = self.data_to_tensor(test)
        print(y_train.shape)
        batch = 8
        x_train_batches = torch.split(x_train, batch)
        # print(len(x_train_batches))
        # print(x_train_batches[0])
        y_train_batches = torch.split(y_train, batch)
        return x_train_batches, y_train_batches, x_test, y_test

    

    def train_model(self, model, function, loss_list, acc_list, mse_loss = True):
        x_train_batches, y_train_batches, x_test, y_test = function
        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)

        for _ in range(200):
            for batch in range(len(x_train_batches)):
                # print(y_train_batches[batch])
                if mse_loss:
                    model.mse_loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
                    loss_list.append(model.mse_loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                else:
                    model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
                    loss_list.append(model.loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                #print(model.mse_loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                
                optimizer.step()  # Perform optimization by adjusting W and b,
                optimizer.zero_grad()  # Clear gradients for next step

        # print(f"Loss: {self.model.loss(x_test, y_test)}")
        acc_list.append(model.mse_acc(x_test, y_test).cpu().detach())
        print(f"Accuracy: {model.mse_acc(x_test, y_test)}")

    def train_model_value(self, model, function, loss_list, acc_list):
        x_train_batches, y_train_batches, x_test, y_test = function
        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)

        for _ in range(200):
            for batch in range(len(x_train_batches)):
                # print(model.loss(x_train_batches[batch], y_train_batches[batch]))
                model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
                loss_list.append(model.loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                optimizer.step()  # Perform optimization by adjusting W and b,
                optimizer.zero_grad()  # Clear gradients for next step


        # print(f"Loss: {self.model.loss(x_test, y_test)}")
        acc_list.append(model.accuracy(x_test, y_test).cpu().detach())
        print(f"Accuracy: {model.accuracy(x_test, y_test)}")

    def get_accuracy(self):
        if len(self.model_accuracy) == 0:
            return 0
        return np.max(self.model_accuracy)

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
        This is a function used to determine if the current player plays with the white or black pieces.
        The black player will always start the game, before the turn alternate between black and white.
        :return: BLACK if move count is an even number, and WHITE if the move count is an odd number
        """
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE

    def train(self, n, mse_loss = True):
        """
        This is the main function for training the models.
        It will use the Monte Carlo Tree to pick the best action, and keep playing until it reaches terminal state.
        Then it backpropogates the result (win, loss, tie) to update the tree.
        The function then:
            -> trains both models with the given data
            -> Saves the policy model and all the training and test data

        :param n: The number of training rounds
        """
        for i in range(n):
            print(f"Training round: {i}")
            # Nullstiller brettet
            self.env.reset()
            done = False
            rounds = 0
            while not done and rounds < 80:
                # Gjør et trekk
                action = self.take_turn_2()
                _, _, done, _ = self.env.step(action)
                print("Round ", rounds)
                rounds += 1

                #self.env.render("terminal")

            #self.backpropagate(self.current_node, self.env.winner())
            self.amountOfSims += 1

            # Iterate the fucking tree
            print("Storing data")
            self.iterate_MCTS(self.R)

            print("Training network")
            #(self.training_data)
            
            self.train_model(self.model, self.get_training_data(self.training_data, self.test_data),
                             self.model_losses, self.model_accuracy, mse_loss)
                             
            self.train_model_value(self.value_model, self.get_training_data(self.training_win, self.test_win),
                            self.value_model_losses, self.value_model_accuracy)

            torch.save(self.model, f"models/SavedModels/{i}_Gen2_{self.model_name}.pt")

            self.R = Node(None, None)
            self.reset()

        #np.save(f"models/training_data/model_{len(self.training_data)}_{uuid.uuid4()}.npy", self.training_data,
                #allow_pickle=True)
        #np.save(f"models/training_data/value_model_{len(self.training_win)}_{uuid.uuid4()}.npy", self.training_win,
                #allow_pickle=True)
        #np.save(f"models/test_data/model_{len(self.test_data)}_{uuid.uuid4()}.npy", self.test_data,
                #allow_pickle=True)
        #np.save(f"models/test_data/value_model_{len(self.test_win)}_{uuid.uuid4()}.npy", self.test_win,
                #allow_pickle=True)

    def opponent_turn_update(self, action):
        self.move_count += 1
        """
        if move in self.current_node.children.keys():
            self.current_node = self.current_node.children[move]
        else:
            new_node = Node(self.current_node, move)
            self.current_node.children.update({(move, new_node)})
            self.current_node = new_node
        """

    def reset(self):
        self.move_count = 0
        self.current_node = self.R

    def print_tree(self):
        """
        This is a function for printing and visualizing the MCTS
        """
        self.R.print_tree()

    def expand_2(self, env_copy):
        """
        """
         # Create all valid children nodes
        valid_moves = env_copy.valid_moves()
        for move in range(len(valid_moves)):
            if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                self.node_count += 1
        #Returns one of the children, it doesn't matter which one because none has been visited
        return self.current_node.children[random.choice(list(self.current_node.children.keys()))]

    def simulate_2(self, env_copy):
        """
        This is a function for simulating a game and return the result.
        If the value mode does not have enough data it will play to terminal state.
        If the value model is good enough it will use the model to predict the winner of the game.
        
        :return: value of the result (win: 1, loss: -1, tie: 0)
        """
    
        # Need to create an environment from self.current_node

        if self.amountOfSims > 9:
            x_tens = torch.tensor(env_copy.state(), dtype=torch.float).to(self.device)
            v = self.value_model.f(x_tens.reshape(-1, 6, self.size, self.size).float()).cpu().detach().item()
            if v > 0.5:
                return 1
            if v < -0.5:
                return -1
            else:
                return 0
        else:
            done = False
            i = 0
            while not done:
                if i > 100:
                    break
                if self.prob_pol:
                    action = self.play_policy_prob(env_copy)
                else:
                    action = self.play_policy_greedy(env_copy)
                state, _, done, _ = env_copy.step(action)
                i+= 1
            v = env_copy.winner()
            return v

    def take_turn_2(self):
        """
        This is a function for picking the best move given a state.
        It will create MCTS tree with 500 iterations, and at the end from the root choose the child with the highest value.
        1.Tree policy: Traversing the Monte Carlo Search Tree to a leaf node, allwaye picking the best child.
        2.Expand the leaf node with all valid children nodes possible, and pick one of them random.
        3.Simulate from the chosen node one time
        4.Backpropogate the result from the simulation all the way back to the root
        
        :return: from root return best action  
        """
        root = self.current_node
        # Create a MCTS from this current state with 500 iterations
        for i in range(500):
            env_copy = copy.deepcopy(self.env)
            #print("iteration", i)
            done = False
            # Tree Policy / Traverse until leaf-node IT IS FUKIN STUCK

            #print("Noden har barn: ",len(self.current_node.children))
            while(len(self.current_node.children) != 0 and not done):
                self.current_node = self.current_node.best_child(self.get_type())
                #print("Noden har barn: ",len(self.current_node.children))
                state, _, done, _ = env_copy.step(self.current_node.action)

            # Expand
            if not done:
                self.current_node = self.expand_2(env_copy)
                state, _, done, _ = env_copy.step(self.current_node.action)

            # Simulate / Rollout
            if not done:
                result = self.simulate_2(env_copy)
            else:
                result = env_copy.winner()
            # Backprop / Traverse back and update each node.
            self.backpropagate(self.current_node, result)

            # Set current node to root
            self.current_node = root

        # From root find the best action
        self.current_node = self.current_node.best_child(self.get_type())
        return self.current_node.action

    def save_data(self):
        # Creating test data and training data for the current node
        if (random.randint(1, 99) <= 20):
            y_t = np.zeros(1)
            if self.current_node.n > 10:
                y_t[0] = self.current_node.V()
                self.test_win.append((self.env.state(), y_t))
            y = self.get_target(self.current_node)
            if np.sum(y) < 1000 and np.sum(y) != 0 and self.current_node.n > 10:
                self.test_data.append((self.env.state(), y))

        else:
            y_t = np.zeros(1)
            if self.current_node.n > 10:
                y_t[0] = self.current_node.V()
                self.training_win.append((self.env.state(), y_t))
            y = self.get_target(self.current_node)

            if np.sum(y) < 1000 and np.sum(y) != 0 and self.current_node.n > 10:
                self.training_data.append((self.env.state(), y))


        """
        
        # Add to children to current node
        for i in self.current_node.children.values():
            env_copy = copy.deepcopy(self.env)
            state, _, _, _ = env_copy.step(i.action)
            y_t = np.zeros(1)
            if not i.n == 0:
                y_t[0] = i.V()
            if (random.randint(1, 99) <= 20):
                y = self.get_target(i)
                if np.sum(y) != 0 and np.sum(y) < 1000:
                    self.test_data.append((state, y))
                if i.n > 5:
                    self.test_win.append((state, y_t))
            else:
                y = self.get_target(i)
                if np.sum(y) != 0 and np.sum(y) < 1000:
                    self.training_data.append((state, y))
                if i.n > 10:
                    self.training_win.append((state, y_t))
        """


    def iterate_MCTS(self, node):

        # save the data
        self.current_node = node
        self.save_data()

        if(len(node.children) == 0):
            return

        for child in node.children.values():
            self.iterate_MCTS(child)

