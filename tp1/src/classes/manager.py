from _typeshed import Self
from state import StateNode
from collections import deque
import sys

from tp1.src.classes.parser import Parser

class ZokobanManager:
    __instance = None

    def __new__(cls, board, goals, player, boxes):
        if cls._instance is None:
            cls._instance = super(ZokobanManager, cls).__new__(cls)
            cls._instance._initialize(board, goals, player, boxes)
        return cls._instance

    def _initialize(self, board, goals, player, boxes):
        self.board = board
        self.goals = goals
        self.root = StateNode(player, boxes)

    @staticmethod
    def getInstance():
        if ZokobanManager.__instance is None:
            raise ValueError("Zokoban Manager hasn't been instantiated")
        return ZokobanManager.__instance

    def bfs(self):
        if self.root is None:
                return

        queue = deque([self.root])

        while queue:
            node = queue.popleft()

            # move in 4 directions...
            # left
            node.player.move()


if __name__ == "__main__":
    if sys.argv[1] is not None:
        board, player, boxes, goals = Parser(sys.argv[1]).parse()
        zokobanManager = ZokobanManager(board, player, boxes, goals)
