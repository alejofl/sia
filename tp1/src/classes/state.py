class State:
    def __init__(self, player, boxes):
        self.player = player
        self.boxes = boxes

    def __hash__(self):
        return hash((self.player, tuple(self.boxes)))