from modelsV2.node import Node, Type
from enum import Enum


class Stage(Enum):
    TRAVERSE = 1
    SIMULATION = 2

class MCTS:

    def __init__(self, env):
        self.moveCount = 0
        self.env = env
        self.R = Node(None, None)
        self.currentNode = self.R
        self.stage = Stage.TRAVERSE
        # Grense for å velge et barn som et godt valg 
        self.traverseLimit = 1
        self.node_count = 0
        # Tree depth
        self.tree_depth = 1
        self.traverseLimit = 4

    def traverse_step(self, type: Type, node: Node):
        if node.best_child(type).get_value(type) > self.traverseLimit:
            self.currentNode = node.best_child(type)
            return self.currentNode.action
        else:
            return self.rollout_step(node)

    # Finner beste barnet til en node, eller lager et nytt med en random action
    def traverse_policy(self, node: Node):
        if len(node.children) != 0:
            # Svart er partall, hvit oddetall
            if self.moveCount % 2 == 0:
                return self.traverse_step(Type.BLACK, node)
            else:
                return self.traverse_step(Type.WHITE, node)
        else:
            # Om ingen barn som oppfyller krav finnes, 
            # lager vi et nytt barn med en random action
            
            return self.rollout_step(node)
                

    def take_turn(self):
        action = -1
        if self.stage is Stage.TRAVERSE:
            action = self.traverse_policy(self.currentNode)
        elif self.stage is Stage.SIMULATION:
            action = self.rollout_policy()
        
        self.moveCount += 1
        return action

    # if there is space for a new node, a new node is added
    def rollout_step(self, node):
        action = self.rollout_policy()
        if action in self.currentNode.children.keys():
            self.currentNode = node.children[action]
        else:
            new_node = Node(node, action)
            node.children.update({(action, new_node)})
            #print(f"action: {action}")
            self.currentNode = new_node
            self.node_count += 1
            self.stage = Stage.SIMULATION
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
