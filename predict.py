import os
import sys
import json
import torch
import joblib
import random
import torch.nn
import pandas as pd
import sklearn.preprocessing

import features

class DynamicPredictor:
    """
    Predict a file's dynamic trace into specified classes using dynamic features model.
    """

    @staticmethod
    def _load_features(featureFile):
        """
        Load the feature list from the specified file.

        :param featureFile: path to file containing features.
        """

        with open(featureFile) as f:
            return json.load(f)

    def __init__(self, filePath, modelPath):
        """
        Initialise the parameters for the model.

        :param filePath: path to the file to be classified.
        :param modelPath: path to the model file.
        """

        self.filePath = filePath
        self.modelPath = modelPath
        self.features = dict()
        self.featureList = self._load_features("features.columns")
        self._generate_features()
        self.model = self._load_model()
        self.le = self._load_labelencoder()
        self.model.eval()

    def _load_labelencoder(self):
        """
        Load the label encodings from a file.
        """

        return joblib.load("encoder.dynamic.le")

    def _generate_features(self):
        """
        Generate features for the given trace.
        """

        APICalls = features.APICalls().processFeatures(self.filePath)
        fileActions = features.FileActions().processFeatures(self.filePath)
        registryActions = features.RegistryActions().processFeatures(self.filePath)
        DLLLoads = features.DLLLoads().processFeatures(self.filePath)

        self.features = {**fileActions, **registryActions, **DLLLoads}
        for call, value in APICalls.items():
            if call in self.featureList:
                self.features[call] = value    
            
        for key in self.featureList:
            if key not in self.features:
                self.features[key] = 0

    def _generate_tensor(self):
        """
        Generate a tensor with the feature dictionary.
        """

        return torch.Tensor(pd.DataFrame(self.features, index=[0]).values)
    
    def _load_model(self):
        """
        Load the model specifications from a file.
        """

        model = DynamicFeaturesModel(len(self.featureList))
        model.load_state_dict(torch.load(self.modelPath))

        return model

    def predict(self):
        """
        Predict the class of the given trace file.
        """

        with torch.no_grad():
            array = self._generate_tensor()
            pred = self.model(array)
            pred = torch.sigmoid(pred)
            pred_tag = torch.round(pred)
            pred_val = pred_tag.cpu().numpy()
            
            return self.le.inverse_transform([int(pred_val[0][0])])[0]

class DynamicFeaturesModel(torch.nn.Module):

    def __init__(self, features):
        super(DynamicFeaturesModel, self).__init__()

        """
        Sequential model for prediction. Initialise the model layers.

        :param features: number of features to train for.
        """
        
        self.layer_1 = torch.nn.Linear(features, 64) 
        self.layer_2 = torch.nn.Linear(64, 32)
        self.layer_3 = torch.nn.Linear(32, 32)
        self.layer_out = torch.nn.Linear(32, 1) 
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)
        self.batchnorm1 = torch.nn.BatchNorm1d(64)
        self.batchnorm2 = torch.nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        """
        Performs a forward pass for the given input.

        :param inputs: set of inputs to use for the forward pass.
        """

        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def Prediction(filePath, modelPath):
    predictor = DynamicPredictor(filePath, modelPath)
    return predictor.predict()

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("usage: python predict.py <trace_file> <model_file>")
        exit()

    print(Prediction(sys.argv[1], sys.argv[2]))