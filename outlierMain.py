import numpy as np
import datetime
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from nilmtk.dataset import DataSet
from collections import Counter
from joblib import parallel_backend
from sklearn.model_selection import KFold
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

## LOAD DATA FROM DATASET ##

def loadApplicationDataFromDataset(application, settings):
    data = DataSet(settings["h5File"])
    df = None

    window = settings["window"]
    if settings["window"] != None:
        data.set_window(window[0],window[1])
    try:
        elecMeter = data.buildings[application[0]].elec
        df = elecMeter[application[1]].power_series_all_data()
    except:
        print(str(e))
        return None

    if df is None:
      return None

    if df.isnull().values.any():
        df.fillna(method='ffill')

    if settings["filterData"] == "rollingMedian":
        df = df.rolling(window=10,
                        center=True,
                        min_periods=1).median()
    elif settings["filterData"] == "hampel":
        df = hampel_filter(df, 10)

    power_values = np.arange(int(max(df))+1).reshape(-1, 1)
    x = df.astype('int').values.reshape(-1, 1)

    return {"power_val": power_values, "x": x}

def loadAggregateDataFromDataset(application, settings):
    data = DataSet(settings["h5File"])
    df = None

    window = settings["window"]
    if settings["window"] != None:
        data.set_window(window[0],window[1])
    try:
        elecMeter = data.buildings[application[0]].elec
        df = elecMeter.mains().power_series_all_data()
    except Exception as e:
        print(str(e))
        return None

    if df is None:
      return None

    if df.isnull().values.any():
        df.fillna(method='ffill')

    if settings["filterData"] == "rollingMedian":
        df = df.rolling(window=10,
                        center=True,
                        min_periods=1).median()
    elif settings["filterData"] == "hampel":
        df = hampelFilter(df, 10)

    power_values = np.arange(int(max(df))+1).reshape(-1, 1)
    x = df.astype('int').values.reshape(-1, 1)

    return {"power_val": power_values, "x": x}

## HAMPEL FILTER ##

def hampelFilter(df, windowSize, nSig=3):
    # https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d
    k = 1.4826 # Gaussian distribution scale factor
    dfFiltered = df.copy()

    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rollMedian = df.rolling(window=2*windowSize,
                            center=True).median()
    rollMAD = k * df.rolling(window=2*windowSize,
                            center=True).apply(MAD, raw=False)
    diff = np.abs(df - rollMedian)

    idx = list(np.argwhere(diff > (nSig * rollMAD)).flatten())
    dfFiltered[idx] = rollMedian[idx]

    return dfFiltered


## ISOLATION FORESTS ###

def calcIsoForest(x_train, x_test):
    try:
        clf = IsolationForest(random_state=0,
                                n_jobs=6).fit(x_train)
        outliers = clf.predict(x_test)
    except:
        return None
    return outliers

## Local Outlier Factor

def calcLOF(x_train, x_test):
    try:
        clf = LocalOutlierFactor(n_neighbors=2,
                                    n_jobs=6)
        outliers = clf.fit_predict(x_test)
    except:
        return None
    return outliers

def storeResults(settings, metrics):
    if os.path.exists(settings["fileName"]):
        fileMode = 'a'
    else:
        fileMode = 'w'

    with open(settings["fileName"], fileMode) as outfile:
        if settings["jsonIndent"]:
            json.dump([settings, metrics],
                        outfile,
                        indent=1)
        else:
            json.dump([settings, metrics], outfile)
        outfile.write("\n")

## METRICS ###

def calcMetrics(results, settings):
    metrics = {}

    if "lof" in settings["algo"]:
        metrics["lof"] = {}
        metrics["lof"]["absApplOut"] = np.count_nonzero(results["outLOFTest"] == -1)
        metrics["lof"]["relApplOut"] = np.count_nonzero(results["outLOFTest"] == -1)/len(results["outLOFTest"])
        lofPowerOutlier = (results["testData"][(np.where(results["outLOFTest"] == -1))])[:,0]
        keys_values = dict(Counter(lofPowerOutlier)).items()
        metrics["lof"]["applOutOccur"] = {str(key): str(value) for key, value in keys_values}
        metrics["lof"]["applOutlier"] = (np.where(results["outLOFTest"] == -1))[0].tolist()

        metrics["lof"]["absPowOut"] = np.count_nonzero(results["outLOFPower"] == -1)
        metrics["lof"]["relPowOut"] = np.count_nonzero(results["outLOFPower"] == -1)/len(results["outLOFPower"])
        metrics["lof"]["powerOutVal"] = (results["powerData"][(np.where(results["outLOFPower"] == -1))])[:,0].tolist()
        metrics["lof"]["powerOutlier"] = (np.where(results["outLOFPower"] == -1))[0].tolist()

    if "if" in settings["algo"]:
        metrics["if"]  = {}
        metrics["if"]["absApplOut"] = np.count_nonzero(results["outIsoForTest"] == -1)
        metrics["if"]["relApplOut"] = np.count_nonzero(results["outIsoForTest"] == -1)/len(results["outIsoForTest"])
        ifPowerOutlier = (results["testData"][(np.where(results["outIsoForTest"] == -1))])[:,0]
        keys_values = dict(Counter(ifPowerOutlier)).items()
        metrics["if"]["applOutOccur"] = {str(key): str(value) for key, value in keys_values}
        metrics["if"]["applOutlier"] = (np.where(results["outIsoForTest"] == -1))[0].tolist()

        metrics["if"]["absPowOut"] = np.count_nonzero(results["outIsoForPower"] == -1)
        metrics["if"]["relPowOut"] = np.count_nonzero(results["outIsoForPower"] == -1)/len(results["outIsoForPower"])
        metrics["if"]["powerOutVal"] = (results["powerData"][(np.where(results["outIsoForPower"] == -1))])[:,0].tolist()
        metrics["if"]["powerOutlier"] = (np.where(results["outIsoForPower"] == -1))[0].tolist()

    return metrics


def calcOutliers(data, settings):
    results = {}
    kf = KFold(n_splits = settings["kFoldSplits"])

    for train_i, test_i in kf.split(data["x"]):
        testStr = str(test_i[0])+" - " +str(test_i[-1])
        settings["testDataStr"] = testStr

        trainStr = str(train_i[0])+" - "+str(train_i[-1])
        settings["trainDataStr"] = trainStr
        results["powerData"] = data["power_val"]

        x_train, x_test  = data["x"][train_i], data["x"][test_i]
        results["testData"] = x_test
        results["trainData"] = x_train
        if "if" in settings["algo"]:
            outliers = calcIsoForest(x_train, x_test)
            if outliers is None:
                return
            results["outIsoForTest"] = outliers

            outliers = calcIsoForest(x_train, data["power_val"])
            if outliers is None:
                return
            results["outIsoForPower"] = outliers

        if "lof" in settings["algo"]:
            outliers = calcLOF(x_train, x_test)
            if outliers is None:
                return
            results["outLOFTest"] = outliers

            outliers = calcLOF(x_train, data["power_val"])
            if outliers is None:
                return
            results["outLOFPower"] = outliers

        metrics = calcMetrics(results, settings)
        storeResults(settings, metrics)

## GET ALL APPLICATIONS ###

def getApplicationsFromDataset(settings, applicationTypes=None):
    applications = []

    data = DataSet(settings["h5File"])

    for building in data.buildings:
            for appl in data.buildings[building].elec.appliances:
                if (building, appl.type["type"]) not in applications:
                    if applicationTypes is None:
                        applications.append((building,
                                            appl.type["type"]))
                    else:
                        if appl.type["type"] in applicationTypes:
                            applications.append((building,
                                                appl.type["type"]))
    return applications

## GET AGGREGATES ###

def getAggregateFromDataset(settings, applicationTypes=None):
    applications = []

    data = DataSet(settings["h5File"])

    for building in data.buildings:
        if (building, None) not in applications:
            applications.append((building, None))

    return applications

## MAIN ###

if __name__ == "__main__":

    ## sys.argv usage - python outlierMain.py
    #                       h5File
    #                       applications
    #                       windowStart
    #                       windowEnd
    ## applications - ["kettle", "computer"] or
    #                       "aggregate" if aggregate data
    #                       should be used
    ## windowStart - "2013-09-08" or None if whole dataset
    #                       should be used

    if len(sys.argv) != 5:
        print("Not enough arguments")
        sys.exit()

    settings = {}
    files = [sys.argv[1]]

    filters = [None, "rollingMedian", "hampel"]
    applicationTypes = sys.argv[2].split(",")

    for filterType in filters: # None, rollingMedian or hampel
        for h5File in files:
            ### SETTINGS ###
            settings["kFoldSplits"] = 5

            if sys.argv[3] == None:
                settings["window"]      = None
            else:
                settings["window"]      = (sys.argv[3], sys.argv[4])

            settings["filterData"]  = filterType
            settings["algo"]        = "if|lof"
            settings["jsonIndent"]  = False # increases readability
            ### /SETTINGS ###

            ## check if directory with h5Files exist else path passed from user
            if os.path.exists("h5Files"):
                settings["h5File"]      = "h5Files/" + h5File
            else:
                settings["h5File"]      = h5File

            ## create directory for results
            if not os.path.exists("simuResults"):
                os.mkdir("simuResults")

            time = datetime.datetime.now().strftime("_%d%m%Y_%H%M%S.json")
            settings["fileName"]    = "simuResults/simRes_" + h5File + time

            if applicationTypes == "aggregate":
                applications = getAggregateFromDataset(settings)
            else:
                applications = getApplicationsFromDataset(settings, applicationTypes)

            for appl in applications:
                if applicationTypes == "aggregate":
                    data = loadAggregateDataFromDataset(appl, settings)
                else:
                    data = loadApplicationDataFromDataset(appl, settings)

                if data is not None:
                    settings["application"] = appl
                    settings["maxPowVal"]   = int(data["power_val"][-1])
                    calcOutliers(data, settings)
