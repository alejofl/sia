from .position import Position
from .elements import Box
from .parser import PLAYER, BOX, BOX_ON_GOAL, GOAL
from .manager import SokobanManager

class Printer:
    @staticmethod    
    def ascii():
        manager = SokobanManager.getInstance()
        
        for i, row in enumerate(manager.board):
            for j, element in enumerate(row):
                position = Position(i, j)
                if manager.root.player.position == position:
                    print(PLAYER, end="")
                elif Box(position) in manager.root.boxes:
                    if position in manager.goals:
                        print(BOX_ON_GOAL, end="")
                    else:
                        print(BOX, end="")
                elif position in manager.goals:
                    print(GOAL, end="")
                else:
                    print(element, end="")
            print()
