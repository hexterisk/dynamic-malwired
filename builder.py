import os
import sys
import json
import jsonlines
import concurrent.futures

import pandas as pd
import features
import config

def Builder(typeClass):
    """
    builder function iterates over all samples and dumps raw features into a json file.
    
    :param typeClass: class of malware to focus on
    """
    print(f"{config.Colours.INFO}[*] Building dataset for {typeClass}.{config.Colours.ENDC}")    

    # Set the path and clear typeClass' download queues and json dump.
    path = f"dataset/{typeClass}"
    try:
        os.remove(f"{path}/dump.jsonl")
    except FileNotFoundError:
        pass
    
    # Define a base feature set for the class.
    featuresSet = set([
        'class',
        'regkey_read',
        'regkey_opened',
        'regkey_deleted',
        'regkey_written',
        "file_failed", 
        "file_copied", 
        "file_exists", 
        "file_opened", 
        "file_read", 
        "file_written",
        "dll_loaded",
    ])

    # Intialize feature class objects.
    APICalls = features.APICalls()
    FileActions = features.FileActions()
    RegistryActions = features.RegistryActions()
    DLLLoads = features.DLLLoads()
    
    # Initialize feature dictionary lists.
    APICallsList = []
    FileActionsList = []
    RegistryActionsList = []
    DLLLoadsList = []

    # Prase files and build feature dictionaries for each feature class.
    for trace in os.listdir(path):
        print(f"[~] Parsing {path}/{trace}")
        APICallsDict = APICalls.processFeatures(f"{path}/{trace}")
        APICallsList.append(APICallsDict)
        # Add all the unique API Calls to the feature set.
        for call, _ in APICallsDict.items():
            featuresSet.add(call)
        FileActionsList.append(FileActions.processFeatures(f"{path}/{trace}"))
        RegistryActionsList.append(RegistryActions.processFeatures(f"{path}/{trace}"))
        DLLLoadsList.append(DLLLoads.processFeatures(f"{path}/{trace}"))

    # Dump all the feature dictionaries for the given typeClass into it's local folder as a json list.
    with jsonlines.open(f"{path}/dump.jsonl", 'w') as buildFile:

        for APICallsDict, FileActionsDict, RegistryActionsDict, DLLLoadsDict in zip(APICallsList, FileActionsList, RegistryActionsList, DLLLoadsList):
            # Merges all the dictionary together into a single dictionary.
            merged = {"class": typeClass, **APICallsDict, **FileActionsDict, **RegistryActionsDict, **DLLLoadsDict}
            for key in featuresSet:
                if key not in merged:
                    merged[key] = 0
            buildFile.write(merged)
            
    print(f"{config.Colours.SUCCESS}[+] Dataset build for {typeClass} complete.{config.Colours.ENDC}")
    return

def Reader(typeClass):
    """
    Reads the dump files for the specified class and returns a pandas dataframe.
    
    :param typeClass: type class for which the data is to be read.
    """
    print(f"{config.Colours.HEADER}[+] Initiated dataset read.{config.Colours.ENDC}")

    # Check if a data dump file for dynamic features exists.
    dumpFile = "dynamic_features.jsonl"
    try:
        os.remove(dumpFile)
    except FileNotFoundError:
        pass
    
    print(f"{config.Colours.INFO}[*] Reading dataset for {typeClass}.{config.Colours.ENDC}")
    path = f"dataset/{typeClass}/dump.jsonl"
    return pd.read_json(path, lines=True)

def BuildDataset():
    """
    Download all the files in a multi-threaded implementation to build a local database.
    """

    print(f"{config.Colours.HEADER}[+] Initiated dataset build.{config.Colours.ENDC}")

    # Multi threaded building process for json dumps.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers = len(config.Classes))
    for typeClass in config.Classes:
        try:
            executor.submit(Builder, typeClass)
            print(f"[+] Thread started for {typeClass}.")
        except:
            print(f"{config.Colours.ERROR}[!] Unable to start thread for {typeClass}.{config.Colours.ENDC}")
    
    # Shutdown the thread manager during exit.
    executor.shutdown(wait=True)
    print(f"{config.Colours.SUCCESS}[+] Dataset build complete.{config.Colours.ENDC}")
    return

if __name__ == "__main__":
    # Shutdown the thread manager during exit.
    BuildDataset()