from collections import deque
from .state import StateNode
from .movements import Movements


class SokobanManager:
    _instance = None

    def __new__(cls, board, goals, player, boxes):
        if cls._instance is None:
            cls._instance = super(SokobanManager, cls).__new__(cls)
            cls._instance._initialize(board, goals, player, boxes)
        return cls._instance

    def _initialize(self, board, goals, player, boxes):
        self.board = board
        self.goals = goals
        self.root = StateNode(player, boxes)
        self.winningPath = deque()

    @staticmethod
    def getInstance():
        if SokobanManager._instance is None:
            raise ValueError("Sokoban Manager hasn't been instantiated")
        return SokobanManager._instance

    def bfs(self):
        if self.root is None:
            return

        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            
            # TODO: Check if winning state
            #       If so, backtrack to get the path and save each node in self.winningPath using append
            # TODO: Check if losing state
            
            upNode = node.clone()
            if upNode.player.canMove(Movements.UP, upNode):
                upNode.player.move(Movements.UP, upNode)
                node.children.append(upNode)
                queue.append(upNode)

            downNode = node.clone()
            if downNode.player.canMove(Movements.DOWN, downNode):
                downNode.player.move(Movements.DOWN, downNode)
                node.children.append(downNode)
                queue.append(downNode)
                
            leftNode = node.clone()
            if leftNode.player.canMove(Movements.LEFT, leftNode):
                leftNode.player.move(Movements.LEFT, leftNode)
                node.children.append(leftNode)
                queue.append(leftNode)
                
            rightNode = node.clone()
            if rightNode.player.canMove(Movements.RIGHT, rightNode):
                rightNode.player.move(Movements.RIGHT, rightNode)
                node.children.append(rightNode)
                queue.append(rightNode)
