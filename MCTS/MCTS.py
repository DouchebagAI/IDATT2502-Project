from MCTS.Node import Node, Type
from enum import Enum
import copy


class Stage(Enum):
    TRAVERSE = 1
    SIMULATION = 2

class MCTS:

    def __init__(self, env):
        self.move_count = 0
        self.env = env
        # Root node for MCTS
        self.R = Node(None, None)
        self.current_node = self.R
        self.stage = Stage.TRAVERSE
        # Number of nodes in the tree 
        self.node_count = 0
    
    # Finds the best child of the current node, 
    # and simulates if the node has never been visited
    def traverse_step(self, type: Type, node: Node):    
        self.current_node = node.best_child(type)
        #Simuler hvis beste verdi har n = 0
        if self.current_node.n == 0:
            self.stage = Stage.SIMULATION
        return self.current_node.action
        

    # Finds the best child of the current node, and creates children if it is a leaf node
    def tree_policy(self, node: Node):
        if len(node.children) != 0:
            # Svart er partall, hvit oddetall
            return self.traverse_step(self.get_type(), node)

        else:
            # Om ingen barn som oppfyller krav finnes, 
            # lager vi et nytt barn med en random action
            
            return self.expand()
                
    # Method to take a sigle turn
    # either via traverse or simulation
    def take_turn(self):
        action = -1
        if self.stage is Stage.TRAVERSE:
            action = self.tree_policy(self.current_node)
        elif self.stage is Stage.SIMULATION:
            action = self.rollout_policy()
        
        self.move_count += 1
        return action

    # If there is space for a new node, a new node is added to the current node
    # returns the action og the new node
    def expand(self):
        # Copy environment
        valid_moves = self.env.valid_moves()
        for index in range(len(valid_moves)):
            if index not in self.current_node.children.keys() and valid_moves[index] == 1.0:
                new_node = Node(self.current_node, index)
                self.current_node.children.update({(index, new_node)})
                self.node_count += 1
                #For each new node, simulate the game and backpropagate the result 20 times
                self.simulate(new_node, 5)
        #print(f"action: {action}")
        self.current_node = self.current_node.best_child(self.get_type())
        self.node_count += 1
        self.stage = Stage.SIMULATION
            # Legg til barn for alle valid moves
        return self.current_node.action

    def get_type(self):
        return Type.BLACK if self.move_count % 2 == 0 else Type.WHITE

    def simulate(self, node: Node, n = 5):
        env_copy = copy.deepcopy(self.env)
        state, reward, done, info = env_copy.step(node.action)
        if done:
            self.backpropagate(node, env_copy.winner())
            return
        original = copy.deepcopy(env_copy)
        for i in range(n):
            env_copy = copy.deepcopy(original)
            done = False
            while not done:
                a = env_copy.uniform_random_action()
                state, reward, done, info = env_copy.step(a)
            self.backpropagate(node, env_copy.winner())

    def rollout_policy(self):
        return self.env.uniform_random_action()

    def backpropagate(self, node: Node, v):
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        node.n += 1
        return node

    def opponent_turn_update(self, action):
        if action in self.current_node.children.keys():
            self.current_node = self.current_node.children[action]
        else:
            new_node = Node(self.current_node, action)
            self.current_node.children.update({(action, new_node)})
            self.current_node = new_node
        
        self.move_count += 1

    def reset(self):
        self.move_count = 0
        self.stage = Stage.TRAVERSE
        self.current_node = self.R

    # Metode for å visualisere treet
    def print_tree(self):
        self.R.print_tree()

