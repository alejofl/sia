import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
import pygame
import numpy as np
from PIL import Image
from .position import Position
from .constants import WALL, PLAYER, BOX, BOX_ON_GOAL, GOAL, PYGAME_SCALE, PYGAME_FPS
from .manager import SokobanManager


class PygameHelper:
    def __init__(self, filename):
        self.filename = filename
        self.frames = []

    def add_frame(self):
        curr_surface = pygame.display.get_surface()
        x3 = pygame.surfarray.array3d(curr_surface)
        x3 = np.moveaxis(x3, 0, 1)
        array = Image.fromarray(np.uint8(x3))
        self.frames.append(array)

    def save(self):
        self.frames[0].save(
            self.filename,
            save_all=True,
            optimize=False,
            append_images=self.frames[1:],
            loop=0,
            duration=300,
        )
        
    @staticmethod
    def load_image(filename):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", filename)
        return pygame.transform.scale(pygame.image.load(path), (PYGAME_SCALE, PYGAME_SCALE))
    
    @staticmethod
    def start_screen(width, height):
        pygame.init()
        screen = pygame.display.set_mode((width * PYGAME_SCALE, height * PYGAME_SCALE))
        clock = pygame.time.Clock()
        return screen, clock
    
    @staticmethod
    def start_frame(screen):
        screen.fill((0, 0, 0))
        
    @staticmethod
    def draw(screen, image, position):
        screen.blit(image, (position.y * PYGAME_SCALE, position.x * PYGAME_SCALE))
        
    @staticmethod
    def end_frame(clock, recorder):
        recorder.add_frame()
        pygame.display.flip()
        clock.tick(60)


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
            
    @staticmethod
    def gif(filename):
        manager = SokobanManager.getInstance()
        path = manager.winningPath.copy()
        
        recorder = PygameHelper(filename)
        screen, clock = PygameHelper.start_screen(len(manager.board[0]), len(manager.board))
        
        box = PygameHelper.load_image("box.png")
        box_on_goal = PygameHelper.load_image("box_on_goal.png")
        goal = PygameHelper.load_image("goal.png")
        player = PygameHelper.load_image("player.png")
        space = PygameHelper.load_image("space.png")
        wall = PygameHelper.load_image("wall.png")
        
        while len(path) > 0:
            node = path.pop()
            
            PygameHelper.start_frame(screen)
            
            for i, row in enumerate(manager.board):
                for j, element in enumerate(row):
                    position = Position(i, j)

                    if node.player.position == position:
                        PygameHelper.draw(screen, player, position)
                    elif position in set(map(lambda box: box.position, node.boxes)):
                        if position in manager.goals:
                            PygameHelper.draw(screen, box_on_goal, position)
                        else:
                            PygameHelper.draw(screen, box, position)
                    elif position in manager.goals:
                        PygameHelper.draw(screen, goal, position)
                    elif element == WALL:
                        PygameHelper.draw(screen, wall, position)
                    else:
                        PygameHelper.draw(screen, space, position)
                
            PygameHelper.end_frame(clock, recorder)
            
        recorder.save()
