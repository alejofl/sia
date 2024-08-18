import json
import sys
from src.classes.parser import Parser
from src.classes.printer import Printer
from src.classes.manager import SokobanManager


if __name__ == "__main__":
    with open(sys.argv[1], "r") as configFile:
        config = json.load(configFile)
        board, player, boxes, goals = Parser(config["board"]).parse()
        
        sm = SokobanManager(board, goals, player, boxes)
        sm.bfs()
        print("Winning path found")
        # Printer.ascii()
        Printer.gif("output.gif")

    sys.exit(0)
