from state import State

class Manager:
    def __init__(self, board, player, boxes, goals):
        self.board = board
        self.goals = goals
        self.currentState = State(player, boxes)

    
        
   