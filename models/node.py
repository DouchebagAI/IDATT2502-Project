import numpy as np
class Node:
    def __init__(self, parent, action, v= 0, n = 0, N = 0, C = 0.95):
        # state = node
        self.action = action
        # Verdien til staten
        self.v = v
        # Antall ganger staten er besøkt
        self.n = n

        # Konstant for å tweake treet
        self.C = C
        # Parent til noden
        self.parent = parent
        # Totalt antall besøk for alle noder på denne dybden
        # Kan vurdere å flytte ut til treet
        if self.is_root():
            self.N = N
        else:
            self.N = parent.n
        # Children
        self.children = {}

    # Formel for å hente verdien til noden (til valg av node i neste)
    def get_value(self):
        return self.v/self.n + self.C * np.sqrt(np.log(self.N)/self.n)

    def is_root(self):
        return self.parent is None

    def update_node(self, v):
        self.n += 1
        self.v = v

    # Finner det barnet med høyest value
    # None hvis ingen barn
    def best_child(self):
        if len(self.children) == 0:
            return None

        index = -1
        for i, item in enumerate(self.children.values()):
            if index != -1 and item.get_value() > self.children[index].get_value():
                index = i

        return self.children[index]