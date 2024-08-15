from abc import ABC, abstractmethod


class Element(ABC):
    def __init__(self, position):
        self.position = position

    @abstractmethod
    def move(self, direction):
        pass
    
    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other):
        return self.position == other.position


class Box(Element):
    def move(self, direction):
        pass
    
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
