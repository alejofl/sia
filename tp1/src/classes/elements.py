from abc import ABC, abstractmethod

from .manager import ZokobanManager
from .movements import Movements
from .position import Position
from .parser import WALL, BOX

class Element(ABC):
    def __init__(self, position):
        self.position = position

    def canMove(self, direction, currentState):
        board = ZokobanManager.getInstance().board
        newPosition = self.position.move(direction)
        return board[newPosition.x][newPosition.y] != WALL

    def move(self, direction, currentState):
        self.position = self.position.move(direction)

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return self.position == other.position


class Box(Element):
    def canMove(self, direction, currentState):
        newPosition = self.position.move(direction)
        return super().canMove(direction, currentState) and Box(newPosition) not in currentState.boxes

    def __str__(self):
        return "Box: " + str(self.position)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return super().__eq__(other)


class Player(Element):
    def canMove(self, direction, currentState):
        if not super().canMove(direction, currentState):
            return False

        newPosition = self.position.move(direction)
        for box in currentState.boxes:
            if box.position == newPosition:
                return box.canMove(direction)
        return True

    def move(self, direction, currentState):
        newPosition = self.position.move(direction)
        for box in currentState.boxes:
            if box.position == newPosition:
                box.move(direction)
                break
        super().move(direction, currentState)

    def __str__(self):
        return "Player: " + str(self.position)

    def __hash__(self):
        return super().__hash__()

    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return super().__eq__(other)
