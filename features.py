import os
import json

class APICalls:
    """
    Feature class for API calls made by the binary.
    """

    def __init__(self):
        pass

    def processFeatures(self, jsonFile):
        """
        Parses the json trace file to fetch total calls, successful calls and failed calls for every API.
        
        :param jsonFile: trace file in json format.
        """
        self.dataDict = {}

        with open(jsonFile) as f:
            data = json.load(f)
            for processName in data["behavior"]["apistats"]:
                for key in data["behavior"]["apistats"][processName].keys():
                    keyDict = key + "_total"
                    if keyDict not in self.dataDict:
                        self.dataDict[keyDict] = data["behavior"]["apistats"][processName][key]
                    else:
                        self.dataDict[keyDict] += data["behavior"]["apistats"][processName][key]

                    for prefix in ["_success", "_fail"]:
                        keyDict = key + prefix
                        if keyDict not in self.dataDict:
                            self.dataDict[keyDict] = 0

            for process in data["behavior"]["processes"]:
                if "calls" in process:
                    for apiCall in process["calls"]:

                        if apiCall["return_value"] == 0:
                            keyDict = apiCall["api"] + "_success"
                        else:
                            keyDict = apiCall["api"] + "_fail"
                        self.dataDict[keyDict] += 1

        return self.dataDict

class FileActions:
    """
    Feature class for file-related actions performed by the binary.
    """

    def __init__(self):
        pass
    
    def processFeatures(self, jsonFile):
        """
        Parses the json trace file to fetch total file actions performed.
        
        :param jsonFile: trace file in json format.
        """
        self.dataDict = {}

        with open(jsonFile) as f:
            data = json.load(f)
            actions = ["file_failed", "file_copied", "file_exists", "file_opened", "file_read", "file_written"]
            for action in actions:
                self.dataDict[action] = 0
            for process in data["behavior"]["generic"]:
                summary = process["summary"]
                for action in actions:
                    if action in summary:
                        self.dataDict[action] = len(summary[action])

        return self.dataDict

class RegistryActions:
    """
    Feature class for registry-related actions performed by the binary.
    """
    
    def __init__(self):
        pass
    
    def processFeatures(self, jsonFile):
        """
        Parses the json trace file to fetch total registry actions performed.
        
        :param jsonFile: trace file in json format.
        """
        self.dataDict = {}

        with open(jsonFile) as f:
            
            data = json.load(f)
            actions = ["regkey_read", "regkey_opened", "regkey_deleted", "regkey_written"]
            for action in actions:
                self.dataDict[action] = 0
            for process in data["behavior"]["generic"]:
                summary = process["summary"]
                for action in actions:
                    if action in summary:
                        self.dataDict[action] = len(summary[action])

        return self.dataDict


class DLLLoads:
    """
    Feature class for the DLLs loaded by the binary.
    """

    def __init__(self):
        pass

    def processFeatures(self, jsonFile):
        """
        Parses the json trace file to fetch the list of DLLs loaded.
        
        :param jsonFile: trace file in json format.
        """
        self.dataDict = {}

        with open(jsonFile) as f:
            data = json.load(f)
            for process in data["behavior"]["generic"]:
                summary = process["summary"]
                if "dll_loaded" in summary:
                    self.dataDict["dll_loaded"] = len(summary["dll_loaded"])
        
        return self.dataDict

class APICallSequences:
    """
    Feature class for the API call sequence signature of the binary.
    """

    def __init__(self):
        """
        Intialise sets of API calls to create a signature.
        """
        
        self.setFile = ["CreateFile", "ReadFile", "WriteFile", "OpenFile", "GetFile", "CopyFile", "FindFile", "DeleteFile", "GetPath", "SearchPath", "CreateDirectory", "GetDirectory", "OpenDirectory", "SetFile", "GetFolder", "MoveFile", "QueryFile", "RemoveDirectory", "NtdeviceIOControlFile"]
        self.A = self.setFile
        self.setSystem = ["NtClose", "NtDelayExecution", "Exception", "GetTime", "GetSystem", "Crypt", "NtQuery", "QuerySystem", "Exit", "Initialize", "CreateInstance", "SetObject", "Hook", "CreateObject", "Debugger", "Service", "Anomaly", "SetError", "Cert", "Privilege", "GetName", "Shellexecute", "Shutdown", "GetObject", "Manager"]
        self.B = self.setSystem
        self.setReg = ["CreateKey", "OpenKey", "CloseKey", "RegGetValue", "RegEnumValue", "RegQuery", "RegEnum", "RegDelete", "RegSet"]
        self.C = self.setReg
        self.setKernel = ["Ldr", "Resource", "Func", "Load", "Uuid", "Hwnd", "Section", "Module", "Dll", "Libm"]
        self.D = self.setKernel
        self.setMemory = ["Memory", "Volume", "Space", "Buffer"]
        self.E = self.setMemory
        self.setProcess = ["Mutant", "OpenProcess", "AssignProcess", "Thread", "SnapShot", "Module", "Process32", "SetUnhandledException", "TerminateProcess", "CreateProcess"]
        self.F = self.setProcess
        self.setWindow = ["GetSystemMetrics", "GetForeGroundWindow", "Console", "KeyState", "Cursor", "RegisterHotKey", "EnumWindows", "SendNotifyMessage", "FindWindow", "CreateCtcCtx", "MessageBox", "State", "Key"]
        self.G = self.setWindow
        self.setNetwork = ["Internet", "Http", "Internal", "WSA", "Adapter", "Host", "DNS", "Addr", "Sock", "Listen", "Recv", "Send", "Select", "Connect", "Bind", "URL", "Interface", "Accept", "NetUser", "NetShare", "NetGet", "Information"]
        self.H = self.setNetwork
        self.setDevice = ["DeviceIOControl", "StdHandle"]
        self.I = self.setDevice
        self.setText = ["String", "Text", "Char"]
        self.J = self.setText

        self.sets = [
            self.setFile,
            self.setSystem,
            self.setReg,
            self.setKernel,
            self.setMemory,
            self.setProcess,
            self.setWindow,
            self.setNetwork,
            self.setDevice,
            self.setText
        ]


    def processFeatures(self, jsonFile):
        """
        Parses the json trace file to fetch the API call sequence signature.
        
        :param jsonFile: trace file in json format.
        """
        
        self.signature = ""

        with open(jsonFile) as f:
            data = json.load(f)
            for process in data["behavior"]["processes"]:
                for call in process["calls"]:
                    for section in self.sets:
                        if any(ele in call["api"] for ele in section):
                            if section == self.A:
                                signature += "A"
                            elif section == self.B:
                                signature += "B"
                            elif section == self.C:
                                signature += "C"
                            elif section == self.D:
                                signature += "D"
                            elif section == self.E:
                                signature += "E"
                            elif section == self.F:
                                signature += "F"
                            elif section == self.G:
                                signature += "G"
                            elif section == self.H:
                                signature += "H"
                            elif section == self.I:
                                signature += "I"
                            elif section == self.J:
                                signature += "J"
                    
        return self.signature
