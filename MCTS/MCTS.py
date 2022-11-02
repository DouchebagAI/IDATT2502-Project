from MCTS.Node import Node, Type
from enum import Enum

class Stage(Enum):
    TRAVERSE = 1
    SIMULATION = 2

class MCTS:

    def __init__(self, env):
        self.moveCount = 0
        self.env = env
        # Root node for MCTS
        self.R = Node(None, None)
        self.currentNode = self.R
        self.stage = Stage.TRAVERSE
        # Number of nodes in the tree 
        self.node_count = 0
    
    # Finds the best child of the current node, 
    # and simulates if the node has never been visited
    def traverse_step(self, type: Type, node: Node):    
        self.currentNode = node.best_child(type)
        #Simuler hvis beste verdi har n = 0
        if self.currentNode.n == 0:
            self.stage = Stage.SIMULATION
        return self.currentNode.action
        

    # Finds the best child of the current node, and creates children if it is a leaf node
    def tree_policy(self, node: Node):
        if len(node.children) != 0:
            # Svart er partall, hvit oddetall
            if self.moveCount % 2 == 0:
                return self.traverse_step(Type.BLACK, node)
            else:
                return self.traverse_step(Type.WHITE, node)
        else:
            # Om ingen barn som oppfyller krav finnes, 
            # lager vi et nytt barn med en random action
            
            return self.expand()
                
    # Method to take a sigle turn
    # either via traverse or simulation
    def take_turn(self):
        action = -1
        if self.stage is Stage.TRAVERSE:
            action = self.tree_policy(self.currentNode)
        elif self.stage is Stage.SIMULATION:
            action = self.rollout_policy()
        
        self.moveCount += 1
        return action

    # If there is space for a new node, a new node is added to the current node
    # returns the action og the new node
    def expand(self):
        action = self.rollout_policy()
        valid_moves = self.env.valid_moves()
        for index in range(len(valid_moves)):
            if index not in self.currentNode.children.keys() and valid_moves[index] == 1.0:
                new_node = Node(self.currentNode, index)
                self.currentNode.children.update({(index, new_node)})
                self.node_count += 1
        #print(f"action: {action}")
        self.currentNode = self.currentNode.children[action]
        self.node_count += 1
        self.stage = Stage.SIMULATION
            # Legg til barn for alle valid moves
                            
        return self.currentNode.action


    def rollout_policy(self):
        return self.env.uniform_random_action()

    def backpropagate(self, node: Node, v):
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        node.n += 1
        self.currentNode = node
        self.moveCount = 0
        self.stage = Stage.TRAVERSE

    def opponent_turn_update(self, action):
        if action in self.currentNode.children.keys():
            self.currentNode = self.currentNode.children[action]
        else:
            new_node = Node(self.currentNode, action)
            self.currentNode.children.update({(action, new_node)})
            self.currentNode = new_node
        
        self.moveCount += 1

    # Metode for å visualisere treet
    def print_tree(self):
        self.R.print_tree()
