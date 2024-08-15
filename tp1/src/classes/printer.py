from .position import Position
from .elements import Box
from .parser import PLAYER, BOX, BOX_ON_GOAL, GOAL


class Printer:
    def __init__(self, board, player, boxes, goals):
        self._board = board
        self._player = player
        self._boxes = boxes
        self._goals = goals
        
    def ascii(self):
        for i, row in enumerate(self._board):
            for j, element in enumerate(row):
                position = Position(i, j)
                if self._player.position == position:
                    print(PLAYER, end="")
                elif Box(position) in self._boxes:
                    if position in self._goals:
                        print(BOX_ON_GOAL, end="")
                    else:
                        print(BOX, end="")
                elif position in self._goals:
                    print(GOAL, end="")
                else:
                    print(element, end="")
            print()
