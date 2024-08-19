import json
import sys
import time
from src.classes.parser import Parser
from src.classes.printer import Printer
from src.classes.manager import SokobanManager


if __name__ == "__main__":
    with open(sys.argv[1], "r") as configFile:
        config = json.load(configFile)
        filenameTemplate = f"{str(int(time.time()))}_{config['algorithm'].replace('*', 'STAR')}{'_' + config['heuristic'] if config['algorithm'] in ('GREEDY', 'A*') else ''}_%s"

        board, player, boxes, goals = Parser(config["board"]).parse()
        
        sm = SokobanManager(board, goals, player, boxes)
        print(f"Running {config['algorithm']} algorithm")
        
        benchmark = sm.run(config["algorithm"], config["heuristic"])
        print("Winning path found!")
        print(benchmark)

        if config["animation"] == "GIF":
            Printer.gif(filenameTemplate % ("animation.gif"))
        elif config["animation"] == "ASCII":
            Printer.ascii(filenameTemplate % ("animation.txt"))

    sys.exit(0)
