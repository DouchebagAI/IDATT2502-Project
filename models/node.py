import numpy as np
class Node:
    def __init__(self, v= 0, n = 0, N = 0, C = 0.1):
        # state = node
        # Verdien til staten
        self.v = v
        # Antall ganger staten er besøkt
        self.n = n
        # Totalt antall besøk for alle noder på denne dybden
        # Kan vurdere å flytte ut til treet
        self.N = N
        # Konstant for å tweake treet
        self.C = C

    def get_value(self):
        return self.v + self.C * np.sqrt(np.log(self.N)/self.n)
    
        