from copy import deepcopy
from collections import deque
from .movements import Movements


class StateNode:
    def __init__(self, player, boxes, parent=None, children=set()):
        self.player = player
        self.boxes = boxes
        self.parent = parent
        self.children = children
        
    def clone(self):
        return StateNode(deepcopy(self.player), {deepcopy(box) for box in self.boxes}, self)
        
    def __hash__(self):
        return hash((self.player, tuple(self.boxes)))
    
    def __str__(self):
        return f"Player: {self.player.position} Boxes: [{', '.join([str(box.position) for box in self.boxes])}]"
    
    def isWinningState(self):
        return all(box.position in SokobanManager.getInstance().goals for box in self.boxes)
    
    def isLosingState(self):
        return self in SokobanManager.getInstance().visitedNodes


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
        self.visitedNodes = set()
        self.winningPath = deque()

    @staticmethod
    def getInstance():
        if SokobanManager._instance is None:
            raise ValueError("Sokoban Manager hasn't been instantiated")
        return SokobanManager._instance
    
    def run(self, algorithm):
        if algorithm == "BFS":
            return self.bfs()
        elif algorithm == "DFS":
            return self.dfs()
        elif algorithm == "GREEDY":
            return self.greedy()
        else:
            raise ValueError("Invalid algorithm")

    def bfs(self):
        if self.root is None:
            return

        queue = deque([self.root])

        movements = 0

        while queue:
            node = queue.popleft()
            print(node)
            
            if node.isWinningState():
                while node is not None:
                    self.winningPath.append(node)
                    node = node.parent
                break
            
            if node.isLosingState():
                continue
            
            self.visitedNodes.add(node)
            movements += 1
            
            upNode = node.clone()
            if upNode.player.canMove(Movements.UP, upNode):
                upNode.player.move(Movements.UP, upNode)
                node.children.add(upNode)
                queue.append(upNode)

            downNode = node.clone()
            if downNode.player.canMove(Movements.DOWN, downNode):
                downNode.player.move(Movements.DOWN, downNode)
                node.children.add(downNode)
                queue.append(downNode)
                
            leftNode = node.clone()
            if leftNode.player.canMove(Movements.LEFT, leftNode):
                leftNode.player.move(Movements.LEFT, leftNode)
                node.children.add(leftNode)
                queue.append(leftNode)
                
            rightNode = node.clone()
            if rightNode.player.canMove(Movements.RIGHT, rightNode):
                rightNode.player.move(Movements.RIGHT, rightNode)
                node.children.add(rightNode)
                queue.append(rightNode)
        return movements

    def dfs(self):
            if self.root is None:
                return

            stack = list([self.root])
            movements = 0
            
            while stack:
                node = stack.pop()
                print(node)
                
                if node.isWinningState():
                    while node is not None:
                        self.winningPath.append(node)
                        node = node.parent
                    break
                
                if node.isLosingState():
                    continue
                
                self.visitedNodes.add(node)
                movements += 1
                
                upNode = node.clone()
                if upNode.player.canMove(Movements.UP, upNode):
                    upNode.player.move(Movements.UP, upNode)
                    node.children.add(upNode)
                    stack.append(upNode)
                    

                downNode = node.clone()
                if downNode.player.canMove(Movements.DOWN, downNode):
                    downNode.player.move(Movements.DOWN, downNode)
                    node.children.add(downNode)
                    stack.append(downNode)
                    
                leftNode = node.clone()
                if leftNode.player.canMove(Movements.LEFT, leftNode):
                    leftNode.player.move(Movements.LEFT, leftNode)
                    node.children.add(leftNode)
                    stack.append(leftNode)
                    
                rightNode = node.clone()
                if rightNode.player.canMove(Movements.RIGHT, rightNode):
                    rightNode.player.move(Movements.RIGHT, rightNode)
                    node.children.add(rightNode)
                    stack.append(rightNode)
            return movements
        

    @staticmethod
    def manhattanHeuristic(self, stateNode):
        cost = 0
        for box in stateNode.boxes:
            minDistance = float("inf") 
            for goal in self.goals:
                distance = abs(box.position.x - goal.x) + abs(box.position.y - goal.y)
                if distance <= minDistance:
                    minDistance = distance
            cost += minDistance
            minDistance = float("inf")
        print(cost)
        return cost

    def greedy(self):

        if self.root is None:
            return

        l = list([self.root])

        movements = 0

        while l:
            node = l.pop(0)
            print(node)
            
            if node.isWinningState():
                while node is not None:
                    self.winningPath.append(node)
                    node = node.parent
                break
            
            if node.isLosingState():
                continue
            
            self.visitedNodes.add(node)
            movements += 1
            
            upNode = node.clone()
            if upNode.player.canMove(Movements.UP, upNode):
                upNode.player.move(Movements.UP, upNode)
                node.children.add(upNode)
                l.append(upNode)

            downNode = node.clone()
            if downNode.player.canMove(Movements.DOWN, downNode):
                downNode.player.move(Movements.DOWN, downNode)
                node.children.add(downNode)
                l.append(downNode)
                
            leftNode = node.clone()
            if leftNode.player.canMove(Movements.LEFT, leftNode):
                leftNode.player.move(Movements.LEFT, leftNode)
                node.children.add(leftNode)
                l.append(leftNode)
                
            rightNode = node.clone()
            if rightNode.player.canMove(Movements.RIGHT, rightNode):
                rightNode.player.move(Movements.RIGHT, rightNode)
                node.children.add(rightNode)
                l.append(rightNode)

            l.sort(reverse=False, key=lambda node: self.manhattanHeuristic(self, node))

        return movements
