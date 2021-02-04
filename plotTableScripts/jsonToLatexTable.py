import os
import shutil
import json

def loadJsonContent(jsonFile):
    with open(jsonFile, 'r') as handle:
        fileContent = handle.readlines()
        fileData = ("".join(fileContent)).replace("\n", "").rstrip()
        json_data = json.loads(fileData)
    return json_data

datFile = ""

try:
    os.mkdir(plotDir)
except:
    pass

for file in os.listdir("./"):
    if ".json" in file:
        datFile = file
        inputData = loadJsonContent(file)

        filters = [None, "rollingMedian", "hampel"]
        tableData = []

        if "aggregate" in datFile:
            tableFile = open("./tables/aggregate_tableData.txt", "w")

            for i in range(1,21):
                tableFile.write("House " + str(i) + " & ")
                for algo in ["if", "lof"]:
                    for filter in filters:
                        for application in inputData:
                            for data in application["data"]:
                                if data["filter"] == filter:
                                    if data["dataset"].split(" ")[1] == str(i):
                                        tableFile.write(str(data[algo]["absApplOut"]) + " & ")
                    if algo == "if":
                        tableFile.write(" & ")
                tableFile.write("     \\\n")
            tableFile.close()
        else:
            for application in inputData:
                tableFile = open("./tables/" + application["application"] + "_appl_tableData.txt", "w")

                for filter in filters:
                    tableFile.write("\n\n--- " + str(filter) + " ---\n\n")
                    for i in range(1,21):
                        for data in application["data"]:
                            if data["filter"] == filter:
                                if data["dataset"].split(" ")[1] == str(i):
                                    tableFile.write("House " + str(i) + " & " + str(data["if"]["absApplOut"]) + " & " + str(data["lof"]["absApplOut"]) + " & & "  + str(data["if"]["relApplOut"]) + " & " + str(data["lof"]["relApplOut"]) + " & & "  + str(data["if"]["relPowerOutValGreater25"]) + " & " + str(data["lof"]["relPowerOutValGreater25"]))
                                    tableFile.write("     \\\n")
                tableFile.close()
