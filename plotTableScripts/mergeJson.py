import os
import sys
import json
import glob

### Merges the json files in the passed directory and stores them in the result file

if len(sys.argv) != 3:
    print("Check argv num: 'directory' 'result_file'")
    exit(0)

def loadJsonContent(jsonFile):
    with open(jsonFile, 'r') as handle:
        json_data = [json.loads(line) for line in handle]
    return json_data

dirPath = os.path.join(sys.argv[1], "*.json")
dirContent = glob.glob(dirPath)
dirContent.sort()

totalList = []

for dir in dirContent:
    if "json" in dir:
        totalList += loadJsonContent(dir)

with open(sys.argv[2], 'w') as f:
    for data in totalList:
        json.dump(data, f)
        f.write("\n")

