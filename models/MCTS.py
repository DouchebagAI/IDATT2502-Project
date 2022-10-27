from models.node import Node
from operator import attrgetter
class MCTS:

    def __init__(self, env):
        self.env = env
        self.R = Node()
        self.currentNode = self.R

    def monte_carlo_tree_search(self):
        while(True):
            self.traverse(self.R)
        pass

    def traverse(self, node: Node):
        pass

    def rollout_policy(self):
        # Finn tilfeldig action ved å ta fra enviroment
        # Ta denne
        pass

    # Finner det barnet med høyest value
    def best_child(self):
        index = -1

        for i, item in enumerate(self.currentNode.children):
            if index != -1 and item.get_value() > self.currentNode.children[index].get_value():
                index = i

        return self.currentNode.children[index]


    def backpropagate(node, result):
        #if is_root(node): return
        #node.stats = update_stats(node, result)
        #backpropagate(node.parent)
        pass
    
    # main function for the Monte Carlo Tree Search
    # def monte_carlo_tree_search(root):
    #     while resources_left(time, computational power):
    #         leaf = traverse(root)
    #         simulation_result = rollout(leaf)
    #         backpropagate(leaf, simulation_result)
            
    #     return best_child(root)

    # # function for node traversal
    # def traverse(node):
    #     while fully_expanded(node):
    #         node = best_uct(node)
            
    #     # in case no children are present / node is terminal
    #     return pick_unvisited(node.children) or node

    # # function for the result of the simulation
    # def rollout(node):
    #     while non_terminal(node):
    #         node = rollout_policy(node)
    #     return result(node)

    # # function for randomly selecting a child node
    # def rollout_policy(node):
    #     return pick_random(node.children)

    # # function for backpropagation
    # def backpropagate(node, result):
    #     if is_root(node) return
    #     node.stats = update_stats(node, result)
    #     backpropagate(node.parent)

    # # function for selecting the best child
    # # node with highest number of visits
    # def best_child(node):
    #     pick child with highest number of visits
