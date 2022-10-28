import json

import numpy as np
class Node:
    def __init__(self, parent, action, v= 0, n = 0, C = 0.95):
        # state = node
        #Action til noden
        self.action = action
        # Verdien til staten
        self.v = v
        # Antall ganger staten/noden er besøkt
        self.n = n
        # Konstant for å tweake treet
        self.C = C
        # Parent til noden
        self.parent = parent
        # Totalt antall besøk for parent

        # Children
        self.children = {}

    # Formel for å hente verdien til noden (til valg av node i neste)
    def get_value(self):
        return self.v/self.n + self.C * np.sqrt(np.log(self.getN())/self.n)

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

    def print_self(self):

        for i in self.children.values():
            print(" - " + str(i.print_self()))
        return self.action

    # Finner det barnet med høyest value
    # None hvis ingen barn
    def best_child(self):
        if len(self.children) == 0:
            return None

        index = -1
        for i, item in enumerate(self.children.values()):
            if index == -1 or item.get_value() > self.children[index].get_value():
                index = item.action

        return self.children[index]
