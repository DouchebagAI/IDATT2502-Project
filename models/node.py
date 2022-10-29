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
    def get_value(self,  type: Type, C=0.95):
        if type == Type.BLACK:
            return self.v / self.n + C * np.sqrt(np.log(self.getN()) / self.n)
        else:
            return (-1)*(self.v / self.n - C * np.sqrt(np.log(self.getN()) / self.n))

    # Gets parents n's
    def getN(self):
        if self.is_root():
            return 0
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

        best_action = list(self.children.values())[0].action
        for node in self.children.values():
            if self.children[best_action].get_value(type) < node.get_value(type):
                best_action = node.action

        return self.children[best_action]

    def __str__(self):
        return f"a: {self.action}, v: {self.v}, n: {self.n}, N: {self.getN()}, parent: {self.parent.action if self.parent != None else 'Root'}"

    def print_node(self, x :int):
        print(self.__str__())
        if len(self.children) == 0:
            return
        print(f"Layer {x}:")
        for i in self.children.values():
            i.print_node(x+1)

    def check_ns(self):
        if len(self.children) == 0:
            return
        sum = 0
        for i in self.children.values():
            i.check_ns()
            sum += i.n
        if sum != self.n:
            print(f"\nSum: {sum}")
            print(f"n: {self.n}\n")
