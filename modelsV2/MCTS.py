from modelsV2.node import Node, Type

class MCTS:

    def __init__(self, env):
        self.moveCount = 0
        self.env = env
        self.R = Node(None, None)
        self.currentNode = self.R
        
        # Grense for å velge et barn som et godt valg 
        self.traverseLimit = 1

    def traverse_step(self, type: Type, node: Node):
        if node.best_child(type).get_value(type) > self.traverseLimit:
            self.currentNode = node.best_child(type)
        else:
            self.rollout_step(node)

    # Finner beste barnet til en node, eller lager et nytt med en random action
    def traverse_policy(self, node: Node):
        if len(node.children) != 0:
            # Svart er partall, hvit oddetall
            if self.moveCount % 2 == 0:
                self.traverse_step(Type.BLACK, node)
            else:
                self.traverse_step(Type.WHITE, node)
        else:
            # Om ingen barn som oppfyller krav finnes, 
            # lager vi et nytt barn med en random action
            self.rollout_step(node)
        self.moveCount += 1

    def rollout_step(self, node):
        action = self.rollout_policy()
        new_node = Node(node, action)
        node.children.update({(action, new_node)})
        self.currentNode = new_node

    def rollout_policy(self):
        return self.env.uniform_random_action()

    def backpropagate(self, node: Node, v):
        while not node.is_root():
            node.update_node(v)
            node = node.parent
        node.n += 1
        self.currentNode = node
        self.moveCount = 0

    def opponent_turn_update(self, action):
        if action in self.currentNode.children.keys():
            self.currentNode = self.currentNode.children[action]
        else:
            new_node = Node(self.currentNode, action)
            self.currentNode.children.update({(action, new_node)})
            self.currentNode = new_node
        
        self.moveCount += 1
