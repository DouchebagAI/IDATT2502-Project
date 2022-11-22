from enum import Enum
import numpy as np


class Type(Enum):
    BLACK = 0
    WHITE = 1


class Node:
    def __init__(self, parent, action, state=[], v=0, n=0):
        self.action = action
        self.state = state
        self.v = v  # The value of the node
        self.n = n  # Number of visits
        self.parent = parent
        self.children = {}

    def get_value(self, type: Type, C=1.4):
        """
        This is a function for retrieving the absolute value of a node.
        The value will be calculated with UCB1.

        :param type: type of the player (black/white)
        :return: Value of a node
        """
        # If node is not visited, return high value to ensure it gets explored
        if self.n is 0:
            return 1000000
        if type == Type.BLACK:
            return self.v / self.n + C * np.sqrt(2 * np.log(self.getN()) / self.n)
        else:
            return (-1) * (self.v / self.n - C * np.sqrt(2 * np.log(self.getN()) / self.n))

    def get_value_default(self, type, C=1.4):
        """
        This is a function for retrieving the default value of a node.
        It will not find the absolute value of a player with the white pieces
        """
        if self.n is 0:
            return 1000000
        if type == Type.BLACK:
            return self.v / self.n + C * np.sqrt(2 * np.log(self.getN()) / self.n)
        else:
            return self.v / self.n - C * np.sqrt(2 * np.log(self.getN()) / self.n)

    def getN(self):
        """
        This is a get-function for the number of visits of the parent node.
        :return: the number of visits of the parent
        """
        if self.is_root():
            return self.n
        return self.parent.n

    def get_type(self):
        """
        This is a get-function for the node type.
        :return: the type of the node (WHITE or BLACK)
        """
        return Type.WHITE if self.parent.state[2][0][0] == 1 else Type.BLACK

    def V(self):
        """
        This is a function for calculating V, by dividing the value with the number of visits.
        :return: V
        """
        return self.v / self.n

    def is_root(self):
        """
        This is a function for
        :return Boolean (True if the node is the root and False if not)
        """
        return self.parent is None

    # Update value and number of visits
    def update_node(self, v):
        """
        This is a function for updating the value and the number of visits of a node.
        It will be called each time the tree backpropogates.

        :param v: (int) The result of the game. (win: 1, tie: 0 or loss: -1)
        """
        self.n = self.n + 1
        self.v = self.v + v

    def best_child(self, type: Type):
        """
        This is a function for finding the child with the highest value

        :param type: the type of the player (black/white)
        :return: (Node) The child with the highest value / None if there are no child nodes
        """
        if len(self.children) == 0:
            return None

        # Find best child
        best_child = max(self.children.values(), key=lambda x: x.get_value(type))

        return best_child

    def __str__(self):
        """
        This is a to string method.
        """
        return f" v: {self.get_value(Type.BLACK).__round__(2)}, n: {self.n}, N: {self.getN()}, a: {self.action}, parent: {self.parent.action if self.parent != None else 'Root'}"

    def print_node(self, depth):
        """
        This is a function for printing a node in the terminal, indented by the depth

        :param depth: The depth of the MCTS
        """
        print(f"{'  ' * 4 * depth}{self}")
        for i in self.children.values():
            i.print_node(depth + 1)

    def print_tree(self):
        """
        This is a function for printing the tree in a nice format
        """
        self.print_node(0)
