import random
import uuid

from models.GoCNN import GoCNN
from models.GoNN import GoNN
from models.CNNValue import GoCNNValue
from MCTSDNN.Node import Node, Type
import torch
import copy
import numpy as np
import gym


class MCTSDNN:

    def __init__(self, env: gym.Env, size, model, kernel_size=3, prob_policy=False, data_augment=False,
                 value_network=True):
        self.value_network = value_network
        self.prob_pol = prob_policy
        self.data_augment = data_augment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.model_name = model
        if model == "Go":
            self.model = GoNN(self.size, kernel_size=kernel_size).to(self.device)
        if model == "Go2":
            self.model = GoCNN(self.size).to(self.device)

        self.value_model = GoCNNValue(self.size, 5).to(self.device)
        self.move_count = 0
        self.env = env
        self.R = Node(None, None)
        self.R.state = self.env.state()
        self.current_node = self.R
        self.node_count = 0
        self.trainingRoundsCompleted = 0

        self.training_win = []
        self.test_win = []

        self.model_losses = []
        self.model_accuracy = []

        self.value_model_losses = []
        self.value_model_accuracy = []

        self.training_data = []
        self.test_data = []

    def take_turn_play(self):
        """
        This is a function for choosing the best move with only the trained neural network model.
        With a greedy- or a probability policy.

        :return: action
        """

        action: int
        if self.prob_pol:
            action = self.play_policy_prob(self.env)
        else:
            action = self.play_policy_greedy(self.env)

        self.move_count += 1
        return action

    def data_augmentation(self, state_and_target):
        """
        Method to create 8 symmetries from a single game state
        Used to create more training data, if needed
        
        :param state_and_target:
        :return: data
        """
        symmetries = self.env.gogame.all_symmetries(state_and_target[0])
        target = state_and_target[1]
        _pass = target[-1]
        target = target[:self.size ** 2].reshape(self.size, self.size)
        data = []

        for i in range(8):
            x = symmetries[i]
            y = target
            if (i >> 0) % 2:
                # Horizontal flip
                x = np.flip(x, 2)
                y = np.fliplr(target)
            if (i >> 1) % 2:
                # Vertical flip
                x = np.flip(x, 1)
                y = np.flipud(target)
            if (i >> 2) % 2:
                # Rotation 90 degrees
                x = np.rot90(x, axes=(1, 2))
                y = np.rot90(target)

            data.append((x, np.append(y, _pass).reshape(26)))

        return data

    def get_target(self, node: Node):
        """
        Method to create a list with actions from a single state
        Used to get target for model to train
        
        :param node: Node
        :return: target of a node
        """
        y_t = np.zeros(self.size ** 2 + 1)
        for i in node.children.values():
            y_t[i.action] = i.get_value_default(self.get_type())

        return y_t

    def action_based_on_prob(self, actions_softmax):
        """
        Method that returns a random action from a list of with actions that are softmaxed
        Creates random number and adds list items value until value is reached. 

        :param actions_softmax:
        :return: action
        """
        numb = np.random.rand()

        sum = 0
        for i, item in enumerate(actions_softmax):
            sum += item
            if numb < sum:
                return i
        return len(actions_softmax)

    def play_policy_prob(self, env):
        """
        Method that returns an action following a probability distributed policy
        
        :param env: The enviorment
        :return: action
        """

        x_tens = torch.tensor(env.state(), dtype=torch.float).to(self.device)
        y = self.model.f_2(x_tens.reshape(-1, 6, self.size, self.size).float())

    
        y = y.cpu().detach()

        y = torch.multiply(y, torch.tensor(env.valid_moves()))
     

        for i, u in enumerate(y[0]):
            if u == 0.0:
                y[0][i] = torch.finfo(torch.float64).min
       

        sm = torch.softmax(torch.tensor(y, dtype=torch.float), dim=1)
        action = self.action_based_on_prob(sm[0])

        return action

    def play_policy_greedy(self, env):
        """
        Method that returns an action following a greedy policy (best move only)
        
        :param env: The enviorment
        """

        x_tens = torch.tensor(env.state(), dtype=torch.float).to(self.device)

        y = self.model.f(x_tens.reshape(-1, 6, self.size, self.size).float())
        index = np.argmax(np.multiply(y.cpu().detach(), env.valid_moves()))

        return index.item()

    def data_to_tensor(self, data):
        """
        This is a function for create a list with states and a list with the matching target
        
        :param data:
        :return a list with states and a list with the matching target
        """
        x_train = []
        random.shuffle(data)
        y_train = []

        for i, tup in enumerate(data):
            x_train.append(tup[0])
            y_train.append(tup[1])

        x_tens = torch.tensor(np.array(x_train), dtype=torch.float).reshape(-1, 6, self.size, self.size).float().to(
            self.device)

        y_tens = torch.tensor(np.array(y_train), dtype=torch.float).float().to(self.device)

        return x_tens, y_tens

    def get_training_data(self, train, test):
        """
        This is a function for getting training data in batches, test data, and training data without batches

        :param train: 
        :param test:
        :return: training data in batches, test data, and training data without batches
        """
        x_train, y_train = self.data_to_tensor(train)
        x_test, y_test = self.data_to_tensor(test)
        print(y_train.shape)
        batch = 8
        x_train_batches = torch.split(x_train, batch)
        y_train_batches = torch.split(y_train, batch)
        
        return x_train_batches, y_train_batches, x_test, y_test, x_train, y_train

    def train_model(self, model, function, loss_list, acc_list, mse_loss=True):
        """
        This is a function for training the policy model
        Using Adam optimizer and training in batches (batch size 8)

        :param model:
        :param function:
        :param loss_list:
        :param acc_list:
        :param mse_loss: boolean
        """
        x_train_batches, y_train_batches, x_test, y_test, x_train, y_train = function
        
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)

        for _ in range(500):
            for batch in range(len(x_train_batches)):

                if mse_loss:
                    model.mse_loss(x_train_batches[batch], y_train_batches[batch]).backward() 
                    loss_list.append(model.mse_loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                else:
                    model.loss(x_train_batches[batch], y_train_batches[batch]).backward()
                    loss_list.append(model.loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())

                optimizer.step()  
                optimizer.zero_grad()  

        acc_list.append(model.mse_acc(x_test, y_test).cpu().detach())
        print(f"Accuracy: {model.mse_acc(x_test, y_test)}")

    def train_model_value(self, model, function, loss_list, acc_list):
        """
        This is a function for training the value model. 
        Using Adam optimizer and training in batches (batch size 8)

        :param model:
        :param function:
        :param loss_list:
        :param acc_list:
        """
        x_train_batches, y_train_batches, x_test, y_test, x_train, y_train = function
        # Optimize: adjust W and b to minimize loss using stochastic gradient descent
        optimizer = torch.optim.Adam(model.parameters(), 0.0001)

        for _ in range(200):
            for batch in range(len(x_train_batches)):
                model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
                loss_list.append(model.loss(x_train_batches[batch], y_train_batches[batch]).cpu().detach())
                optimizer.step()  # Perform optimization by adjusting W and b,
                optimizer.zero_grad()  # Clear gradients for next step

        acc_list.append(model.accuracy(x_test, y_test).cpu().detach())
        print(f"Accuracy: {model.accuracy(x_test, y_test)}")

    def get_type(self):
        """
        This is a function used to determine if the current player plays with the white or black pieces.
        The black player will always start the game, before the turn alternate between black and white.

        :return: BLACK if move count is an even number, and WHITE if the move count is an odd number
        """
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE

    def train(self, n, mse_loss=True):
        """
        This is the main function for training the models.
        It will use the Monte Carlo Tree to pick the best action, and keep playing until it reaches terminal state.
        Then it backpropogates the result (win, loss, tie) to update the tree.
        The function then:
            -> trains both models with the given data
            -> Saves the policy model and all the training and test data

        :param n: The number of training rounds
        :param mse_loss: Boolean
        """
        for i in range(n):
            print(f"Training round: {i}")
            # Resets the board
            self.env.reset()
            done = False
            rounds = 0
            while not done and rounds < 80:
                # Make a move
                action = self.take_turn()
                _, _, done, _ = self.env.step(action)

                print("Round ", rounds)
                rounds += 1

            self.trainingRoundsCompleted += 1

            # Iterate the tree and store data
            self.iterate_MCTS(self.R)

            # Training models
            self.train_model(self.model, self.get_training_data(self.training_data, self.test_data),
                             self.model_losses, self.model_accuracy, mse_loss)

            self.train_model_value(self.value_model, self.get_training_data(self.training_win, self.test_win),
                                   self.value_model_losses, self.value_model_accuracy)

            torch.save(self.model, f"models/SavedModels/{i}_Gen2_{self.model_name}.pt")
            torch.save(self.value_model, f"models/SavedModels/{i}_Gen2_{self.model_name}_value.pt")
            self.env.reset()
            self.R = Node(None, None)
            self.reset()

        # Uncomment if wanting to save training and test data

        # np.save(f"models/training_data/model_{len(self.training_data)}_{uuid.uuid4()}.npy", self.training_data,
        # allow_pickle=True)
        # np.save(f"models/training_data/value_model_{len(self.training_win)}_{uuid.uuid4()}.npy", self.training_win,
        # allow_pickle=True)
        # np.save(f"models/test_data/model_{len(self.test_data)}_{uuid.uuid4()}.npy", self.test_data,
        # allow_pickle=True)
        # np.save(f"models/test_data/value_model_{len(self.test_win)}_{uuid.uuid4()}.npy", self.test_win,
        # allow_pickle=True)

    def opponent_turn_update(self, action):
        """
        This is a function for updating the move count and the node.
        The player needs to know which move the opponent did.

        :action: The action the opponent did
        """
        self.move_count += 1

        if action in self.current_node.children.keys():
            self.current_node = self.current_node.children[action]
        else:
            new_node = Node(self.current_node, action)
            self.current_node.children.update({(action, new_node)})
            self.current_node = new_node

    def reset(self):
        """
        This is a function for reseting MCTS
        """
        self.move_count = 0
        self.current_node = self.R
        self.R.state = self.env.state()

    def print_tree(self):
        """
        This is a function for printing and visualizing the MCTS
        """
        self.R.print_tree()

    def take_turn(self):
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
        # Create MCTS from this current state with 500 iterations
        for i in range(250):
            env_copy = copy.deepcopy(self.env)
            # print("iteration", i)
            done = False
            # Tree Policy / Traverse until leaf-node

            # print("Noden har barn: ",len(self.current_node.children))
            while (len(self.current_node.children) != 0 and not done):
                self.current_node = self.current_node.best_child(self.get_type())
                # print("Noden har barn: ",len(self.current_node.children))
                state, _, done, _ = env_copy.step(self.current_node.action)

            # Expand
            if not done:
                self.current_node = self.expand(env_copy)
                state, _, done, _ = env_copy.step(self.current_node.action)

            # Simulate / Rollout
            if not done:
                result = self.simulate(env_copy)
            else:
                result = env_copy.winner()
            # Backprop / Traverse back and update each node.
            self.backpropagate(self.current_node, result)

            # Set current node to root
            self.current_node = root

        # From root find the best action
        self.current_node = self.current_node.best_child(self.get_type())
        return self.current_node.action

    def expand(self, env_copy):
        """
        This is a function for expanding the MCTS with all 
        valid children nodes depending on the state.

        :return: a random node of the created nodes 
        """
        # Create all valid children nodes
        valid_moves = env_copy.valid_moves()
        for move in range(len(valid_moves)):
            if move not in self.current_node.children.keys() and valid_moves[move] == 1.0:
                new_node = Node(self.current_node, move)
                self.current_node.children.update({(move, new_node)})
                new_node.state = env_copy.gogame.next_state(env_copy.state(), move)
                self.node_count += 1
        # Returns one of the children, it doesn't matter which one because none has been visited
        return self.current_node.children[random.choice(list(self.current_node.children.keys()))]

    def simulate(self, env_copy):
        """
        This is a function for simulating a game and return the result.
        If the value mode does not have enough data it will play to terminal state.
        If the value model is good enough it will use the model to predict the winner of the game.
        
        :return: value of the result (win: 1, loss: -1, tie: 0)
        """

        if self.value_network and (self.trainingRoundsCompleted > 2 or (len(self.value_model_accuracy) != 0 and
                                                                        self.value_model_accuracy[-1] > 0.7)):
            print("Using value network")
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
                i += 1
            v = env_copy.winner()
            return v

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

    def save_data(self):
        """
        This is a function for saving the node to test and training data.
        -> 20% of the time append the data to test data
        -> 80% of the time append the data to training data
        Also checks if the node has been visited enough time to be stored
        """

        self.move_count = int(self.current_node.state[2][0][0])
        # Creating test data and training data for the current node
        if random.randint(1, 99) <= 20:
            y_t = np.zeros(1)
            if self.current_node.n > 10:
                y_t[0] = self.current_node.V()
                self.test_win.append((self.current_node.state, y_t))
            y = self.get_target(self.current_node)
            if np.sum(y) < 1000 and np.sum(y) != 0 and self.current_node.n >= 30:
                if self.data_augment:
                    self.test_data.extend(self.data_augmentation((self.current_node.state, y)))
                else:
                    self.test_data.append((self.current_node.state, y))

        else:
            y_t = np.zeros(1)
            if self.current_node.n > 10:
                y_t[0] = self.current_node.V()
                self.training_win.append((self.current_node.state, y_t))
            y = self.get_target(self.current_node)

            if np.sum(y) < 1000 and np.sum(y) != 0 and self.current_node.n >= 30:
                if self.data_augment:
                    self.training_data.extend((self.data_augmentation((self.current_node.state, y))))
                else:
                    self.training_data.append((self.current_node.state, y))

    def iterate_MCTS(self, node):
        """
        This is a function for iterating the MCTS and on the way
        save each of the node
        
        :param node: The current node to be saved
        """

        self.current_node = node
        self.save_data()

        if len(node.children) == 0:
            return

        for child in node.children.values():
            self.iterate_MCTS(child)
