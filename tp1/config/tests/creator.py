import json

maxLevel = 6

boards = []
for level in range(1, maxLevel + 1):
    boards.append(f"soko{level:02}")
    
uninformed = ["DFS", "BFS"]
informed = ["GREEDY", "A*"]
heuristics = ["MANHATTAN", "EUCLIDEAN", "BOUNDING_BOX"]

for board in boards:
    for algorithm in uninformed:
        config = {
            "board": f"config/boards/{board}.txt",
            "algorithm": algorithm,
            "heuristic": "",
            "animation": "GIF",
            "reps": 10
        }
        file = open(f"config/config_{board}_{algorithm}.json", "w")
        file.write(json.dumps(config))
        file.close()
    for algorithm in informed:
        for heuristic in heuristics:
            config = {
                "board": f"config/boards/{board}.txt",
                "algorithm": algorithm,
                "heuristic": heuristic,
                "animation": "GIF",
                "reps": 10
            }
            file = open(f"config/config_{board}_{algorithm}_{heuristic}.json", "w")
            file.write(json.dumps(config))
            file.close()
