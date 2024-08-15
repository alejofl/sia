from .elements import Box, Player
from .position import Position


WALL = "#"
BOX = "$"
BOX_ON_GOAL = "*"
GOAL = "."
PLAYER = "@"
EMPTY = " "


class Parser:
    def __init__(self, boardPath):
        self._boardPath = boardPath
    
    def parse(self):
        matrix = []
        player = None
        boxes = set()
        goals = set()

        rows = []
        with open(self._boardPath, "r") as board:
            rows = board.readlines()

        max_length = max([len(row) - 1 for row in rows])

        for i, row in enumerate(rows):
            new_row = []
            for j, char in enumerate(row[:-1]):
                position = Position(i, j)
                element_for_matrix = EMPTY
                if char == PLAYER:
                    player = Player(position)
                elif char == BOX:
                    boxes.add(Box(position))
                elif char == BOX_ON_GOAL:
                    boxes.add(Box(position))
                    goals.add(position)
                    element_for_matrix = GOAL
                elif char == GOAL:
                    goals.add(position)
                    element_for_matrix = GOAL
                elif char == WALL:
                    element_for_matrix = WALL

                new_row.append(element_for_matrix)
            new_row += [EMPTY] * (max_length - len(new_row))
            matrix.append(new_row)

        return matrix, player, boxes, goals
