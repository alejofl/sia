import json
import sys
from src.classes.parser import Parser
from src.classes.printer import Printer


if __name__ == "__main__":
    with open(sys.argv[1], "r") as configFile:
        config = json.load(configFile)
        board, player, boxes, goals = Parser(config["board"]).parse()
        
        Printer(board, player, boxes, goals).ascii()

    sys.exit(0)
