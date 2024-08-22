import json
import sys
import time
import csv
from src.classes.parser import Parser
from src.classes.printer import Printer
from src.classes.manager import SokobanManager


if __name__ == "__main__":
    with open(sys.argv[1], "r") as configFile:
        config = json.load(configFile)
        filenameTemplate = f"{str(int(time.time()))}_{config['algorithm'].replace('*', 'STAR')}{'_' + config['heuristic'] if config['algorithm'] in ('GREEDY', 'A*') else ''}_%s"

        output = open(filenameTemplate % ("results.csv"), "w", newline="")
        writer = csv.DictWriter(output, fieldnames=None)

        print(f"Running {config['algorithm']} algorithm {config['reps']} times.")
        for i in range(0, config["reps"]):
            board, player, boxes, goals = Parser(config["board"]).parse()

            sm = SokobanManager(board, goals, player, boxes)

            benchmark = sm.run(config["algorithm"], config["heuristic"])

            if i == 0:
                writer.fieldnames = benchmark.keys()
                writer.writeheader()
            writer.writerow(benchmark)

            print(f"Rep {i+1}/{config['reps']} finished.")
            if i == 0:
                if config["animation"] == "GIF":
                    Printer.gif(filenameTemplate % ("animation.gif"))
                elif config["animation"] == "ASCII":
                    Printer.ascii(filenameTemplate % ("animation.txt"))

        output.close()

    sys.exit(0)
