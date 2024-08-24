import math
import time
from copy import deepcopy
from collections import deque
from queue import PriorityQueue
from .movements import Movements


class StateNode:
    def __init__(self, player, boxes, parent=None, children=set()):
        self.player = player
        self.boxes = boxes
        self.parent = parent
        self.children = children
        
    def clone(self):
        return StateNode(deepcopy(self.player), {deepcopy(box) for box in self.boxes}, self)
    
    def __eq__(self, other):
        return self.player == other.player and self.boxes == other.boxes
        
    def __hash__(self):
        return hash((self.player, tuple(self.boxes)))
    
    def __str__(self):
        return f"Player: {self.player.position} Boxes: [{', '.join([str(box.position) for box in self.boxes])}]"

    def __lt__(self, other):
        # This method is only implemented for the PriorityQueue to work.
        # Semantically, StateNodes are not comparable, so this method should not be used.
        return 0
    
    def isWinningState(self):
        return all(box.position in SokobanManager.getInstance().goals for box in self.boxes)
    
    def isLosingState(self):
        return self in SokobanManager.getInstance().visitedNodes
    
    def manhattanDistance(self):
        manager = SokobanManager.getInstance()
        cost = 0
        
        for box in self.boxes:
            minDistance = float("inf")
            for goal in manager.goals:
                distance = abs(box.position.x - goal.x) + abs(box.position.y - goal.y)
                if distance < minDistance:
                    minDistance = distance
            cost += minDistance
        return cost
    
    def euclideanDistance(self):
        manager = SokobanManager.getInstance()
        cost = 0
        
        for box in self.boxes:
            minDistance = float("inf")
            for goal in manager.goals:
                distance = math.sqrt((box.position.x - goal.x) ** 2 + (box.position.y - goal.y) ** 2)
                if distance < minDistance:
                    minDistance = distance
            cost += minDistance
        return cost

    def boundingBox(self):
        manager = SokobanManager.getInstance()
        cost = 0
        
        for box in self.boxes:
            minDistance = float("inf")
            for goal in manager.goals:
                distance =  abs(box.position.x - goal.x) + abs(box.position.y - goal.y)
                if distance < minDistance:
                    minDistance = distance
                
            if minDistance > cost:
                cost = minDistance
        return cost


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
    
    def _getHeuristic(self, heuristic):
        heuristicFn = None
        if heuristic == "MANHATTAN":
            heuristicFn = lambda node: node.manhattanDistance()
        elif heuristic == "EUCLIDEAN":
            heuristicFn = lambda node: node.euclideanDistance()
        elif heuristic == "BOUNDING_BOX":
            heuristicFn = lambda node: node.boundingBox()
        else:
            raise ValueError("Invalid heuristic")
        return heuristicFn
    
    def startBenchmark(self):
        self.visitedNodes = set()
        self.winningPath = deque()
        self.borderNodesCount = 0
        self.startTime = time.time()
        
    def endBenchmark(self):
        return {
            "cost": len(self.winningPath),
            "expandedNodes": len(self.visitedNodes),
            "borderNodes": self.borderNodesCount,
            "executionTime": time.time() - self.startTime
        }
    
    def run(self, algorithm, heuristic=None):
        self.startBenchmark()
        if algorithm == "BFS":
            self.bfs()
        elif algorithm == "DFS":
            self.dfs()
        elif algorithm == "GREEDY":
            self.greedy(self._getHeuristic(heuristic))
        elif algorithm == "A*":
            self.aStar(self._getHeuristic(heuristic))
        else:
            raise ValueError("Invalid algorithm")
        return self.endBenchmark()

    def bfs(self):
        if self.root is None:
            return

        queue = deque([self.root])

        while queue:
            node = queue.popleft()
            
            if node.isWinningState():
                while node is not None:
                    self.winningPath.append(node)
                    node = node.parent
                break
            
            if node.isLosingState():
                continue
            
            self.visitedNodes.add(node)
            
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

        self.borderNodesCount = len(queue)

    def dfs(self):
        if self.root is None:
            return

        stack = list([self.root])
            
        while stack:
            node = stack.pop()
            
            if node.isWinningState():
                while node is not None:
                    self.winningPath.append(node)
                    node = node.parent
                break
            
            if node.isLosingState():
                continue
            
            self.visitedNodes.add(node)
            
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

        self.borderNodesCount = len(stack)

    def aStar(self, heuristic):
        if self.root is None:
            return

        queue = PriorityQueue()
        queue.put((0, 0, self.root))

        while not queue.empty():
            _, g, node = queue.get()
            
            if node.isWinningState():
                while node is not None:
                    self.winningPath.append(node)
                    node = node.parent
                break
            
            if node.isLosingState():
                continue
            
            self.visitedNodes.add(node)
            
            upNode = node.clone()
            if upNode.player.canMove(Movements.UP, upNode):
                upNode.player.move(Movements.UP, upNode)
                node.children.add(upNode)
                queue.put((g + 1 + heuristic(upNode), g + 1, upNode))

            downNode = node.clone()
            if downNode.player.canMove(Movements.DOWN, downNode):
                downNode.player.move(Movements.DOWN, downNode)
                node.children.add(downNode)
                queue.put((g + 1 + heuristic(downNode), g + 1, downNode))
                
            leftNode = node.clone()
            if leftNode.player.canMove(Movements.LEFT, leftNode):
                leftNode.player.move(Movements.LEFT, leftNode)
                node.children.add(leftNode)
                queue.put((g + 1 + heuristic(leftNode), g + 1, leftNode))
                
            rightNode = node.clone()
            if rightNode.player.canMove(Movements.RIGHT, rightNode):
                rightNode.player.move(Movements.RIGHT, rightNode)
                node.children.add(rightNode)
                queue.put((g + 1 + heuristic(rightNode), g + 1, rightNode))
        
        self.borderNodesCount = queue.qsize()

    def greedy(self, heuristic):
        if self.root is None:
            return

        queue = PriorityQueue()
        queue.put((0, self.root))

        while not queue.empty():
            _, node = queue.get()
            
            if node.isWinningState():
                while node is not None:
                    self.winningPath.append(node)
                    node = node.parent
                break
            
            if node.isLosingState():
                continue
            
            self.visitedNodes.add(node)
            
            upNode = node.clone()
            if upNode.player.canMove(Movements.UP, upNode):
                upNode.player.move(Movements.UP, upNode)
                node.children.add(upNode)
                queue.put((heuristic(upNode), upNode))

            downNode = node.clone()
            if downNode.player.canMove(Movements.DOWN, downNode):
                downNode.player.move(Movements.DOWN, downNode)
                node.children.add(downNode)
                queue.put((heuristic(downNode), downNode))
                
            leftNode = node.clone()
            if leftNode.player.canMove(Movements.LEFT, leftNode):
                leftNode.player.move(Movements.LEFT, leftNode)
                node.children.add(leftNode)
                queue.put((heuristic(leftNode), leftNode))
                
            rightNode = node.clone()
            if rightNode.player.canMove(Movements.RIGHT, rightNode):
                rightNode.player.move(Movements.RIGHT, rightNode)
                node.children.add(rightNode)
                queue.put((heuristic(rightNode), rightNode))

        self.borderNodesCount = queue.qsize()
