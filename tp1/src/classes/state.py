class StateNode:
    def __init__(self, player, boxes, left = None, right = None, up = None, down = None):
        self.player = player
        self.boxes = boxes
        self.left = left
        self.right = right
        self.up = up
        self.down = down

    def __hash__(self):
        return hash((self.player, tuple(self.boxes)))
