from enum import Enum

import numpy as np


class Type(Enum):
    BLACK = 0
    WHITE = 1

class Node:
    def __init__(self, parent, action, v=0, n=0):
        # state = node
        # Action til noden
        self.action = action
        # Verdien til staten
        self.v = v
        # Antall ganger staten/noden er besøkt
        self.n = n
        # Parent til noden
        self.parent = parent
        # Totalt antall besøk for parent
        # Children
        self.children = {}

    # Formel for å hente verdien til noden (til valg av node i neste)
    def get_value(self,  type: Type, C=1.4):
        
        #if type is Type.BLACK:
            #print("\nBlack")
        #else:
            #print("\nWhite")
        
        #print(f"v/n: {self.v / self.n}")
        #print(f"explo: {C * np.sqrt(np.log(self.getN()) / self.n)}")
        #print(f"explo 2: {np.sqrt(np.log(self.getN()) / self.n)}")
        
        if type == Type.BLACK:
            #print(f"Val: {self.v / self.n + C * np.sqrt(np.log(self.getN()) / self.n)}")
            return self.v / self.n + C * np.sqrt(np.log(self.getN()) / self.n)
        else:
            #print(f"Val: {(-1)*(self.v / self.n - C * np.sqrt(np.log(self.getN()) / self.n))}")
            return (-1)*(self.v / self.n - C * np.sqrt(np.log(self.getN()) / self.n))

    # Gets parents n's
    def getN(self):
        if self.is_root():
            return self.n
        else:
            return self.parent.n

    def is_root(self):
        return self.parent is None

    # Oppdaterer verdiene
    def update_node(self, v):
        self.n = self.n + 1
        self.v = self.v + v

    # Finner det barnet med høyest value
    # None hvis ingen barn
    def best_child(self, type: Type):
        if len(self.children) == 0:
            return None

        # Find best child
        best_child = max(self.children.values(), key=lambda x: x.get_value(type))

        return best_child

    def __str__(self):
        return f" v: {self.get_value(Type.BLACK).__round__(2)}, n: {self.n}, N: {self.getN()}, a: {self.action}, parent: {self.parent.action if self.parent != None else 'Root'}"

    # Prints the tree in a nice format
    def print_node(self, depth):
        print(f"{'  '*4*depth}{self}")
        for i in self.children.values():
            i.print_node(depth+1)

    def check_ns(self):
        if len(self.children) == 0:
            return
        sum = 0
        for i in self.children.values():
            i.check_ns()
            sum += i.n
        if sum != self.n:
            print("\nError")
            print(f"Sum: {sum}")
            print(f"n: {self.n}\n")

    def print_tree(self):
        self.print_node(0)