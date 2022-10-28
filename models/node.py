import json

import numpy as np


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
    def get_value(self, C=0.95, y=1):
        return y * (self.v / self.n + C * np.sqrt(np.log(self.getN()) / self.n))

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
        self.n += 1
        self.v = v

    # Finner det barnet med høyest value
    # None hvis ingen barn
    def best_child(self, y=1):
        if len(self.children) == 0:
            return None

        index = -1
        for i, item in enumerate(self.children.values()):
            if index == -1 or item.get_value(y) > self.children[index].get_value(y):
                index = item.action

        return self.children[index]
