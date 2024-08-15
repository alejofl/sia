from abc import ABC, abstractmethod
from .movements import Movements
from .position import Position
from .parser import WALL, BOX

class Element(ABC):
    def __init__(self, position):
        self.position = position

    def canMove(self, manager, direction):
        board = manager.board
        newPosition = self.position.move(direction)
        return board[newPosition.x][newPosition.y] != WALL and Box(newPosition) not in manager.currentState.boxes

    @abstractmethod
    def move(self, direction):
        pass
    
    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return self.position == other.position


class Box(Element):
    def move(self, direction):
        match direction:
            case Movements.UP:
                self.position = Position(self.position.x, self.position.y + 1)
            case Movements.DOWN:
                self.position = Position(self.position.x, self.position.y - 1)
            case Movements.LEFT:
                self.position = Position(self.position.x - 1, self.position.y)
            case Movements.RIGHT:
                self.position = Position(self.position.x + 1, self.position.y)
    
    def __str__(self):
        return "Box: " + str(self.position)
    
    def __hash__(self):
        return super().__hash__()
    
    def __eq__(self, other):
        if not isinstance(other, Box):
            return False
        return super().__eq__(other)


class Player(Element):
    def move(self, direction):
        pass
    
    def __str__(self):
        return "Player: " + str(self.position)
    
    def __hash__(self):
        return super().__hash__()
    
    def __eq__(self, other):
        if not isinstance(other, Player):
            return False
        return super().__eq__(other)
