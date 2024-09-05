from jsonschema import validate as json_validate
import sys
import json


if __name__ == "__main__":
    with open("config/schema.json", "r") as schemaFile, open(sys.argv[1], "r") as configFile:
        schema = json.load(schemaFile)
        config = json.load(configFile)
        json_validate(config, schema)

        print("Config file is valid.")

    sys.exit(0)
