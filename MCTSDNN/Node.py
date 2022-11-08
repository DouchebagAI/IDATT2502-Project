from enum import Enum
import numpy as np

class Type(Enum):
    BLACK = 0
    WHITE = 1

class Node:
    def __init__(self, parent, action,state = [], v=0, n=0):
        self.action = action
        # The value of the node
        self.state = state
        self.v = v
        # Number of visits
        self.n = n
        self.parent = parent
        self.children = {}

    # Gets the absolute value of the node
    def get_value(self,  type: Type, C=1.4):
        #If node is not visited, return high value to ensure it gets explored
        if self.n is 0:
            return 1000000
        if type == Type.BLACK:
            return self.v / self.n + C * np.sqrt(2*np.log(self.getN()) / self.n)
        else:
            return (-1)*(self.v / self.n - C * np.sqrt(2*np.log(self.getN()) / self.n))

    def get_value_default(self, type, C = 1.4):
        if type == Type.BLACK:
            return self.v / self.n + C * np.sqrt(2*np.log(self.getN()) / self.n)
        else:
            return (self.v / self.n - C * np.sqrt(2*np.log(self.getN()) / self.n))

    # Retuns the number of visists of the parent
    def getN(self):
        if self.is_root():
            return self.n
        p = self.parent
        while not p.is_root():
            p = self.parent
        return p.n
        

    def is_root(self):
        return self.parent is None

    # Update value and number of visits
    def update_node(self, v):
        self.n = self.n + 1
        self.v = self.v + v

    # Finds the child with the highest value and returns none if no children
    def best_child(self, type: Type):
        if len(self.children) == 0:
            return None

        # Find best child
        best_child = max(self.children.values(), key=lambda x: x.get_value(type))

        return best_child

    # To string method
    def __str__(self):
        return f" v: {self.get_value(Type.BLACK).__round__(2)}, n: {self.n}, N: {self.getN()}, a: {self.action}, parent: {self.parent.action if self.parent != None else 'Root'}"

    # Prints a node, indendet by the depth
    def print_node(self, depth):
        print(f"{'  '*4*depth}{self}")
        for i in self.children.values():
            i.print_node(depth+1)

    # Prints the tree in a nice format
    def print_tree(self):
        self.print_node(0)