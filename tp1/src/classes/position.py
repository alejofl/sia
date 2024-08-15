from .movements import Movements


class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def move(self, movement):
        match movement:
            case Movements.UP:
                return Position(self.x, self.y + 1)
            case Movements.DOWN:
                return Position(self.x, self.y - 1)
            case Movements.LEFT:
                return Position(self.x - 1, self.y)
            case Movements.RIGHT:
                return Position(self.x + 1, self.y)
            case _:
                return self
