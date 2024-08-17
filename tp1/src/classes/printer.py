from .position import Position
from .elements import Box
from .constants import PLAYER, BOX, BOX_ON_GOAL, GOAL
from .manager import SokobanManager

class Printer:
    @staticmethod    
    def ascii():
        manager = SokobanManager.getInstance()
        path = manager.winningPath.copy()
        while len(path) > 0:
            node = path.pop()
            Printer._asciiNode(node)
            print()
            print()

    @staticmethod
    def _asciiNode(node):
        manager = SokobanManager.getInstance()
        
        for i, row in enumerate(manager.board):
            for j, element in enumerate(row):
                position = Position(i, j)

                if node.player.position == position:
                    print(PLAYER, end="")
                elif position in set(map(lambda box: box.position, node.boxes)):
                    if position in manager.goals:
                        print(BOX_ON_GOAL, end="")
                    else:
                        print(BOX, end="")
                elif position in manager.goals:
                    print(GOAL, end="")
                else:
                    print(element, end="")
            print()
