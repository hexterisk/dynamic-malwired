import os
import json
import torch
import joblib
import sklearn
import numpy as np
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection

import config
import builder

# Dataloader for training data.
class TrainData(torch.utils.data.Dataset):
    
    def __init__(self, X_data, Y_data):
        """
        Set the train data and labels for the loader.

        :param X_data: train data.
        :param y_data: train labels.
        """

        self.X_data = X_data
        self.Y_data = Y_data
        
    def __getitem__(self, index):
        """
        Fetch an object from the dataset.

        :param index: index of the object to be returned from the dataset.
        """

        return self.X_data[index], self.Y_data[index]
        
    def __len__ (self):
        """
        Return samples in the dataset.
        """

        return len(self.X_data)

# Dataloader for testing data.
class TestData(torch.utils.data.Dataset):
    
    def __init__(self, X_data):
        """
        Set the test data for the loader.

        :param X_data: train data.
        """

        self.X_data = X_data
        
    def __getitem__(self, index):
        """
        Fetch an object from the dataset.

        :param index: index of the object to be returned from the dataset.
        """

        return self.X_data[index]
        
    def __len__ (self):
        """
        Return samples in the dataset.
        """

        return len(self.X_data)
    
class BinaryClassificationModel(torch.nn.Module):
    
    def __init__(self):
        """
        Create a model.
        """
        super(BinaryClassificationModel, self).__init__()
    
        self.layer_1 = torch.nn.Linear(X.shape[1], 64) 
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

def BinaryAccuracy(y_pred, y_test):
    """
    Returns the accuracy given a set of predictions and labels.
    
    :param y_pred: predictions list.
    :param y_test: labels list.
    """

    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

if __name__ == "__main__":

    # Set the hyper-parameters.
    EPOCHS = 50
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001



    ### Prepare data.

    # Read the json dumps for each type class and create a unified dataframe.
    dynamic_features_df = pd.DataFrame()
    for typeClass in config.Classes:
        dynamic_features_df = dynamic_features_df.append(builder.Reader(typeClass))
    # Replace all the NaNs with 0.
    dynamic_features_df = dynamic_features_df.fillna(0)
    
    # Drop multiple classes, keep binary classes.
    # dynamic_features_df.loc[dynamic_features_df["class"] != "Benign", "class"] = "Malware"

    # Dump the unified dataframe into a file to be used later during prediction.
    with open("features.columns", 'w') as f:
        # Skip the "class" column(the first one).
        f.write(json.dumps(dynamic_features_df.columns.values.tolist()[1:]))

    # Segregate the data from the dataframe.
    X = dynamic_features_df.drop(["class"], axis=1).values
    # Scale the data before training.
    X = sklearn.preprocessing.StandardScaler().fit_transform(X)
    
    # Create a label encoder object.
    labelEncoder = sklearn.preprocessing.LabelEncoder()
    # Segregate the labels from the dataframe.
    Y = dynamic_features_df["class"].values
    # Encode the labels.
    Y = labelEncoder.fit_transform(Y)
    # Dump the labels
    joblib.dump(labelEncoder, "encoder.dynamic.le")

    # Split the dataset for training and testing.
    X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)

    # Instantiate custom dataloaders with the data.
    trainData = TrainData(torch.FloatTensor(X_train), torch.FloatTensor(Y_train))
    testData = TestData(torch.FloatTensor(X_test))

    # Instantiate the dataloaders for training and testing with the parameters.
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=1)



    ### Prepare the model.

    # Set the training device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Instantiate the model.
    model = BinaryClassificationModel()
    model.to(device)
    print(model)

    # Set the parameters for backward pass.
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)



    ### Train the model.

    model.train()
    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in trainLoader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = BinaryAccuracy(y_pred, y_batch.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            

        print(f"Epoch {e+0:03}: | Loss: {epoch_loss/len(trainLoader):.5f} | Acc: {epoch_acc/len(trainLoader):.3f}")



    ### Evaluate the model.

    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in testLoader:
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]



    ### Analyse the results.
    
    # Get the sklearn report.
    report = sklearn.metrics.classification_report(Y_test, y_pred_list)
    print(report)

    # Get the f-score.
    fScore = sklearn.metrics.fbeta_score(Y_test, y_pred_list, beta=1.0, average="micro")
    print(f"F-Score: {fScore}")

    # Save the model for use in predicition.
    torch.save(model.state_dict(), "model.mdl")