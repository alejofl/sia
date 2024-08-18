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
        print(f'Running {config["algorithm"]} algorithm')
        sm.run(config["algorithm"], config["heuristic"])
        print(f'Winning path found! ({len(sm.winningPath)} movements)')
        # Printer.ascii()
        Printer.gif("output.gif")

    sys.exit(0)
