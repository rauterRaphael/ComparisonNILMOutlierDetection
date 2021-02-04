import os
import sys
import csv
import json
import numpy as np
import copy
import glob
import shutil
import collections
import matplotlib.pyplot as plt

### ------------------------------------------------------------- ###

def getJsonFile():
    if not os.path.exists("simuResults"):
        return input("Path to json file: ")
    else:
        i = 1
        dirContent = os.listdir("simuResults")
        dirContent = glob.glob("simuResults/*.json")
        dirContent.sort()
        for dir in dirContent:
            if "json" in dir:
                print(str(i) + " - " + str(dir).split("/")[1])
                i += 1
        for j in range(3):
            fileNum = int(input("File num: ")) - 1
            if fileNum in range(i-1):
                filePath = dirContent[fileNum]
                return filePath
            else:
                print("Computer sagt nein.")
    return None

def loadJsonContent(jsonFile):
    with open(jsonFile, 'r') as handle:
        json_data = [json.loads(line) for line in handle]
    return json_data

### ------------------------------------------------------------- ###

def getMaxBuildingNum(fileContent):
    maxNum = 0

    for line in fileContent:
        buildNum = line[0]["application"][0]
        if buildNum > maxNum:
            maxNum = buildNum

    return maxNum

def getAllApplications(fileContent):
    applicationsAll = []
    applicationsMerged = []

    for line in fileContent:
        appl = line[0]["application"]
        if appl not in applicationsAll:
            applicationsAll.append(appl)
            if appl[1] not in applicationsMerged:
                applicationsMerged.append(appl[1])

    return applicationsAll, applicationsMerged

### ------------------------------------------------------------- ###

def storeResults(fileName, outlier):
    if os.path.exists(fileName):
        fileMode = 'a'
    else:
        fileMode = 'w'

    with open(fileName, fileMode) as outfile:
        json.dump(outlier, outfile, indent=1)

        outfile.write("\n")


def getOutPerAppl(fileContent, applications):
    outPerAppl = []
    applData   = {}
    kFoldSplits = 5

    for filterType in [None, "rollingMedian", "hampel"]:
        for appl in applications:
            applData["application"] = appl
            applData["filter"] = filterType
            applData["settings"] = {}
            index = 0
            for idx, line in enumerate(fileContent):
                settings = line[0]
                metrics  = line[1]
                if metrics:
                    if settings["application"] == appl and applData["filter"] == settings["filterData"]:
                        applData["settings"] = settings
                        if metrics["if"]:
                            if "if" not in applData:
                                applData["if"] = {}
                                applData["if"]["absApplOut"] = 0
                                applData["if"]["relApplOut"] = 0
                                applData["if"]["absPowOut"]  = 0
                                applData["if"]["relPowOut"]  = 0

                            applData["if"]["absApplOut"] += metrics["if"]["absApplOut"]
                            applData["if"]["relApplOut"] += metrics["if"]["relApplOut"]
                            applData["if"]["absPowOut"]  += metrics["if"]["absPowOut"]
                            applData["if"]["relPowOut"]  += metrics["if"]["relPowOut"]

                        if metrics["lof"]:
                            if "lof" not in applData: 
                                applData["lof"] = {}
                                applData["lof"]["absApplOut"] = 0
                                applData["lof"]["relApplOut"] = 0
                                applData["lof"]["absPowOut"]  = 0
                                applData["lof"]["relPowOut"]  = 0

                            applData["lof"]["absApplOut"] += metrics["lof"]["absApplOut"]
                            applData["lof"]["relApplOut"] += metrics["lof"]["relApplOut"]
                            applData["lof"]["absPowOut"]  += metrics["lof"]["absPowOut"]
                            applData["lof"]["relPowOut"]  += metrics["lof"]["relPowOut"]

            if applData["if"]:
                applData["if"]["relApplOut"] /= kFoldSplits
                applData["if"]["relPowOut"]  /= kFoldSplits

            if applData["lof"]:
                applData["lof"]["relApplOut"] /= kFoldSplits
                applData["lof"]["relPowOut"]  /= kFoldSplits

            applData["calc"] = {}
            applData["calc"] = getOutlierPerAppl(fileContent, appl, filterType)
           
            outPerAppl.append(copy.deepcopy(applData))
            applData = {}

    return outPerAppl

def getOutlierPerAppl(fileContent, appl, filterType):
    applData   = {}
    applData["application"] = appl
    applData["settings"] = {}
    for line in fileContent:
        settings = line[0]
        metrics  = line[1]
        if metrics:
            if settings["application"] == appl and settings["filterData"] == filterType:
                applData["settings"] = settings
                applData["dataLen"] = 0
                if int(settings["testDataStr"].split("- ")[1]) > applData["dataLen"]:
                    applData["dataLen"] = int(settings["testDataStr"].split("- ")[1])

                if "if" in metrics:
                    indexOffset = int(settings["testDataStr"].split(" -")[0])
                    if "if" not in applData: 
                        applData["if"] = {}
                        applData["if"]["applOutOccur"] = preprocessOutlierOccurDict(metrics["if"]["applOutOccur"])
                        applData["if"]["outlierIndices"] = []
                        applData["if"]["powerOutVal"] = []
                    else:
                        currOccur = preprocessOutlierOccurDict(metrics["if"]["applOutOccur"])
                        applData["if"]["applOutOccur"] = {x: applData["if"]["applOutOccur"].get(x, 0) + currOccur.get(x, 0) 
                                                         for x in set(applData["if"]["applOutOccur"]).union(currOccur)} 
                    applData["if"]["outlierIndices"] += [x+indexOffset for x in metrics["if"]["applOutlier"]]
                    for val in metrics["if"]["powerOutVal"]:
                        if val not in applData["if"]["powerOutVal"]:
                            applData["if"]["powerOutVal"].append(val)

                if "lof" in metrics:
                    indexOffset = int(settings["testDataStr"].split(" -")[0])
                    if "lof" not in applData: 
                        applData["lof"] = {}
                        applData["lof"]["applOutOccur"] = preprocessOutlierOccurDict(metrics["lof"]["applOutOccur"])
                        applData["lof"]["outlierIndices"] = []
                        applData["lof"]["powerOutVal"] = []
                    else:
                        currOccur = preprocessOutlierOccurDict(metrics["lof"]["applOutOccur"])
                        applData["lof"]["applOutOccur"] = {x: applData["lof"]["applOutOccur"].get(x, 0) + currOccur.get(x, 0) 
                                                          for x in set(applData["lof"]["applOutOccur"]).union(currOccur)}
                    applData["lof"]["outlierIndices"] += [x+indexOffset for x in metrics["lof"]["applOutlier"]] 
                    for val in metrics["lof"]["powerOutVal"]:
                        if val not in applData["lof"]["powerOutVal"]:
                            applData["lof"]["powerOutVal"].append(val)

    if "if" in applData:
        total = 0
        for item in applData["if"]["applOutOccur"].items():

            if item[0] > 25:
                total += item[1]

        applData["if"]["powerOutValGreater25"] = total / applData["dataLen"]

    if "lof" in applData:
        total = 0
        for item in applData["lof"]["applOutOccur"].items():
            if item[0] > 25:
                total += item[1]
        applData["lof"]["powerOutValGreater25"] = total / applData["dataLen"]
    return applData

### ------------------------------------------------------------- ###

def createTablesFromMetrics(fileContent):

    settings = fileContent[0][0]
    h5FileName  = (settings["h5File"].split("/")[1]).split(".")[0]
    #csvFileName = "simuResults/tableData_" + (settings["fileName"].split("simRes_")[1]).split(".json")[0] + ".json"
    csvFileName = input("Filename: ")
    applicationsAll, applicationsMerged  = getAllApplications(fileContent)

    outlierPerAppl = getOutPerAppl(fileContent, applicationsAll)

    toCSV = []
    col = {}
    entry = {}

    for appl in applicationsMerged:
        col["application"] = appl
        col["data"] = []

        for outAppl in outlierPerAppl:
            if appl == outAppl["application"][1]:
                entry["dataset"] = h5FileName + " " + str(outAppl["application"][0])
                entry["filter"]  = outAppl["filter"]
                if outAppl["if"]:
                    entry["if"] = {}
                    entry["if"]["absApplOut"] = round(outAppl["if"]["absApplOut"], 2)
                    entry["if"]["relApplOut"] = round(outAppl["if"]["relApplOut"], 2)
                    entry["if"]["absPowOut"]  = round(outAppl["if"]["absPowOut"], 2)
                    entry["if"]["relPowOut"]  = round(outAppl["if"]["relPowOut"], 2)
                    entry["if"]["relPowerOutValGreater25"] = round(outAppl["calc"]["if"]["powerOutValGreater25"], 2)

                if outAppl["lof"]:
                    entry["lof"] = {}
                    entry["lof"]["absApplOut"] = round(outAppl["lof"]["absApplOut"], 2)
                    entry["lof"]["relApplOut"] = round(outAppl["lof"]["relApplOut"], 2)
                    entry["lof"]["absPowOut"]  = round(outAppl["lof"]["absPowOut"], 2)
                    entry["lof"]["relPowOut"]  = round(outAppl["lof"]["relPowOut"], 2)
                    entry["lof"]["relPowerOutValGreater25"] = round(outAppl["calc"]["lof"]["powerOutValGreater25"], 2)
                col["data"].append(copy.deepcopy(entry))

        toCSV.append(copy.deepcopy(col))
    try:
        print("Saving...")
        print(csvFileName)
        os.remove(csvFileName)
    except OSError:
        pass

    with open(csvFileName, "w") as outfile:
        json.dump(toCSV, outfile, indent=1)


### ------------------------------------------------------------- ###

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

### ------------------------------------------------------------- ###

def preprocessOutlierOccurDict(applOutOccur):
    outlierOccur = list(applOutOccur.items())
    outlierOccur = [tuple(map(int, value)) for value in outlierOccur]
    outlierOccur.sort(key=lambda x: x[0])
    outlierOccurDict = {}
    for val in outlierOccur:
        outlierOccurDict[val[0]] = val[1]
    return outlierOccurDict

### ------------------------------------------------------------- ###

def createOutlierOccurHistogram():
    figFileName = text["plotDir"] + "/" + text["algo"] + "_" + text["h5FileName"] + str(text["building"]) + "_hist_" + str(text["application"]) + ".png"
    figureTxt = "UOD using " + text["algo"] + " - " + text["h5FileName"] + " " + str(text["building"]) + " - Histogram - " + str(text["application"])
    plt.figure(figureTxt)
    plt.bar(range(len(outlierOccur)), list(outlierOccur.values()), align='center')
    plt.title(figureTxt)
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")
    plt.savefig(figFileName)
    plt.close()

def createAlgoComparison(data, text, outlierIndices):
    figFileName = text["plotDir"] + "/" + text["algo"] + "_" + text["h5FileName"] + str(text["building"]) + "_diff_" + str(text["application"]) + ".png"
    figureTxt = "UOD using " + text["algo"] + " - " + text["h5FileName"] + " " + str(text["building"]) + " - Comparison - " + str(text["application"])
    plt.figure(figureTxt)
    plt.subplot(2,1,1)
    plt.plot(data["x"], label= "data")
    plt.subplot(2,1,2)
    plt.plot(outlierIndices, data["x"][outlierIndices], label="outlier")
    plt.title(figureTxt)
    plt.xlabel("Time in s")
    plt.ylabel("Power Values in W")
    plt.legend(loc="upper left")
    plt.savefig(figFileName)
    plt.close()

def createHistoComparison(outlier, text):
    figFileName = text["plotDir"] + "/" + text["h5FileName"] + str(text["building"]) + "_" + str(text["application"]) + "-" + str(text["filter"]) +  ".png"

    plt.figure()

    plt.tight_layout()
    plt.subplot(2,2,1)
    plt.title("Histogram - IF outliers")
    plt.bar(range(len(outlier["if"]["applOutOccur"])), list(outlier["if"]["applOutOccur"].values()), align='center', label="IF", color="orange")
    plt.legend(loc="upper left")
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")
    #plt.plot()

    plt.subplot(2,2,2)
    plt.title("Histogram - LOF outliers")
    plt.bar(range(len(outlier["lof"]["applOutOccur"])), list(outlier["lof"]["applOutOccur"].values()), align='center', label="LOF", color="blue")
    plt.legend(loc="upper left")
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")
    #plt.plot()

    similarIndices = []
    for item in outlier["lof"]["outlierIndices"]:
        if item in outlier["if"]["outlierIndices"]:
            similarIndices.append(item)
    for item in outlier["if"]["outlierIndices"]:
        if item in outlier["lof"]["outlierIndices"] and item not in similarIndices:
            similarIndices.append(item)

    plt.subplot(2,2,3)
    plt.title("Histogram - IF & LOF outliers")
    similarPowVal = {}
    for lofOutK, lofOutI in outlier["lof"]["applOutOccur"].items():
        for ifOutK, ifOutI in outlier["if"]["applOutOccur"].items():
            if lofOutK == ifOutK:
                if lofOutI > ifOutI:
                    similarPowVal[lofOutK] = ifOutI
                else:
                    similarPowVal[lofOutK] = lofOutI

    plt.bar(range(len(similarPowVal)), list(similarPowVal.values()), align='center', label="SIM", color="red")
    plt.legend(loc="upper left")
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")

    plt.subplot(2,2,4)
    columns = ('total sim.', "rel. sim. IF", "rel. sim. LOF")
    plt.axis("off")
    cell_text = [[str(len(similarIndices)), str(round(len(similarIndices)/len(outlier["if"]["outlierIndices"]), 4)), str(round(len(similarIndices)/len(outlier["lof"]["outlierIndices"]), 4))]]
    plt.table(cellText=cell_text,
                      colLabels=columns,
                      cellLoc="center",
                      loc="center"
    )
    plt.tight_layout()
    plt.savefig(figFileName)
    plt.close()

def createHistoComparisonAgg(outlier, text):
    figFileName = text["plotDir"] + "/" + text["h5FileName"] + str(text["building"]) + "_" + str(text["application"]) + "-" + str(text["filter"]) +  ".png"

    plt.figure()

    plt.tight_layout()
    plt.subplot(2,2,1)
    plt.title("Histogram - IF outliers")
    plt.bar(range(len(outlier["if"]["applOutOccur"])), list(outlier["if"]["applOutOccur"].values()), align='center', label="IF", color="orange")
    plt.legend(loc="upper left")
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")
    #plt.plot()

    plt.subplot(2,2,2)
    plt.title("Histogram - LOF outliers")
    plt.bar(range(len(outlier["lof"]["applOutOccur"])), list(outlier["lof"]["applOutOccur"].values()), align='center', label="LOF", color="blue")
    plt.legend(loc="upper left")
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")
    #plt.plot()

    similarIndices = []
    for item in outlier["lof"]["outlierIndices"]:
        if item in outlier["if"]["outlierIndices"]:
            similarIndices.append(item)
    for item in outlier["if"]["outlierIndices"]:
        if item in outlier["lof"]["outlierIndices"] and item not in similarIndices:
            similarIndices.append(item)

    plt.subplot(2,2,3)
    plt.title("Histogram - IF & LOF outliers")
    similarPowVal = {}
    for lofOutK, lofOutI in outlier["lof"]["applOutOccur"].items():
        for ifOutK, ifOutI in outlier["if"]["applOutOccur"].items():
            if lofOutK == ifOutK:
                if lofOutI > ifOutI:
                    similarPowVal[lofOutK] = ifOutI
                else:
                    similarPowVal[lofOutK] = lofOutI

    plt.bar(range(len(similarPowVal)), list(similarPowVal.values()), align='center', label="SIM", color="red")
    plt.legend(loc="upper left")
    plt.xlabel("Power Values in W")
    plt.ylabel("# Counts")

    plt.subplot(2,2,4)
    columns = ('total sim.', "rel. sim. IF", "rel. sim. LOF")
    plt.axis("off")
    cell_text = [[str(len(similarIndices)), str(round(len(similarIndices)/len(outlier["if"]["outlierIndices"]), 4)), str(round(len(similarIndices)/len(outlier["lof"]["outlierIndices"]), 4))]]
    plt.table(cellText=cell_text,
                      colLabels=columns,
                      cellLoc="center",
                      loc="center"
    )
    plt.tight_layout()
    plt.savefig(figFileName)
    plt.close()

### ------------------------------------------------------------- ###

def createPlotsFromMetric(fileContent):
    plotDir = "simuResults/plots" # + (fileContent[0][0]["fileName"].split("simRes_")[1]).split(".json")[0]

    if "aggre" in fileContent[0][0]["fileName"]: plotDir = "simuResults/plotsAgg"
    h5FileName  = (fileContent[0][0]["h5File"].split("/")[1]).split(".")[0]

    try:
        shutil.rmtree(plotDir)
    except OSError:
        pass
    os.mkdir(plotDir)

    applicationsAll, applicationsMerged  = getAllApplications(fileContent)

    if "aggregate" in fileContent[0][0]["fileName"]:
        i = 0
        for appl in applicationsAll:
            printProgressBar(i, (len(applicationsAll))-1, prefix = 'Plotting:', suffix = 'Complete', length = 50)
            figFileName = plotDir + "/" + h5FileName + "_" + str(appl) +  ".png"

            plt.figure()

            for idx, filterType in enumerate([None, "rollingMedian", "hampel"]):
                outlier = getOutlierPerAppl(fileContent, appl, filterType)

                if idx == 0:
                    colour = "orange"
                elif idx == 1:
                    colour = "blue"
                else:
                    colour = "red"

                plt.subplot(2,3,idx+1)
                plt.title("IF - " + str(filterType))
                
                plt.bar(range(len(outlier["if"]["applOutOccur"])), list(outlier["if"]["applOutOccur"].values()), align='center', label="IF", color=colour)
                plt.legend(loc="upper left")
                plt.xlabel("Power Values in W")
                plt.ylabel("# Counts")

                plt.subplot(2,3,3+idx+1)
                plt.title("LOF - " + str(filterType))
                plt.bar(range(len(outlier["lof"]["applOutOccur"])), list(outlier["lof"]["applOutOccur"].values()), align='center', label="LOF", color=colour)
                plt.legend(loc="upper left")
                plt.xlabel("Power Values in W")
                plt.ylabel("# Counts")
                
            i+=1
            plt.tight_layout()
            plt.savefig(figFileName)
            plt.close()
            

    else:
        i = 0
        for filterType in [None, "rollingMedian", "hampel"]:
            i=0
            for appl in applicationsAll:
                outlier = getOutlierPerAppl(fileContent, appl, filterType)
                printProgressBar(i, (len(applicationsAll))-1, prefix = 'Plotting:', suffix = 'Complete', length = 50)
                text = {}
                text["h5FileName"] = h5FileName
                text["plotDir"]    = plotDir
                text["application"] = outlier["application"][1]
                text["building"]    = outlier["application"][0]
                text["filter"]     = filterType

                createHistoComparison(outlier, text)

                i+=1


### ------------------------------------------------------------- ###

if __name__ == "__main__":

    jsonFile = getJsonFile()

    if jsonFile is not None:
        fileContent = loadJsonContent(jsonFile)
        createTablesFromMetrics(fileContent)
        createPlotsFromMetric(fileContent)

