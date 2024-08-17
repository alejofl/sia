from copy import deepcopy

class StateNode:
    def __init__(self, player, boxes, parent=None, children=set()):
        self.player = player
        self.boxes = boxes
        self.parent = parent
        self.children = children
        
    def clone(self):
        return StateNode(deepcopy(self.player), deepcopy(self.boxes), self)
        
    def __hash__(self):
        return hash((self.player, tuple(self.boxes)))
