# Dynamic Malwired
Classifies if a PE's dynamic analysis report from Cuckoo sandbox amongst the specified classes.
The given trace file can be categorized among the classes defined in the config file. The user must modify the classes based on the dataset and the requirement. Simply edit the classes in [config.py](https://github.com/hexterisk/dynamic-malwired/blob/master/config.py) and the model in [train.py](https://github.com/hexterisk/dynamic-malwired/blob/master/train.py) as per requirement. The currently configured model specs gives a 97% accuracy on a dataset of decent size.

Also checkout the [repository](https://github.com/hexterisk/static-malwired/) hosting the POC for a classifier working on static analysis.

DISCLAIMER: The whole suite has been created, although the user would have to acquire the dataset and train the model by themselves.

## Algorithm

A [blog-post](https://hexterisk.github.io/blog/posts/2020/08/04/classification-of-malwares-through-dynamic-analysis/) on the features used in this POC.

A dynamic analysis report from Cuckoo sandbox is taken as input to extract features corresponding to the binary's behaviour. Different sets of features can be tried out on different models for the best results. A neural network using all the features has been trained as a POC with a good accuracy.

The API Call Sequence feature can be independently used for behavior fingerprinting. The APIs can be divided into different sets to create a signature and then a language model can be used on a dataset of signatures.

## Code Base

The dataset I used was confidential and I cannot disclose it. The user must generate/procure a dataset on their own.

### Flow:

1. Setup a `dataset` folder full of Cuckoo sandbox reports in json format in the following structure:
```bash
dynamic-malwired/
├── dataset/
│   ├── classA
│   │   ├── jsonReportA
│   │   ├── jsonReportB
│   │   .
│   │   .
│   │   .
│   │   └── jsonReportZ
│   ├── classB
│   │   ├── jsonReportA
│   │   .
│   │   └── jsonReportZ
│   ├── classC
│   │   ├── jsonReportA
│   │   .
│   │   └── jsonReportZ
.   .
.   .
.   .
│   └── classZ
├── app.py
├── builder.py
.
.
.
└── train.py
```

2. Run [builder.py](https://github.com/hexterisk/dynamic-malwired/blob/master/builder.py) to build the dataset from the given database.

3. Run [train.py](https://github.com/hexterisk/dynamic-malwired/blob/master/train.py) to train a model.

4. Run [predict.py](https://github.com/hexterisk/dynamic-malwired/blob/master/predict.py) to predict a sample's type.

### Components:

[app.py](https://github.com/hexterisk/dynamic-malwired/blob/master/app.py): Flask RESTful API on which the project has been deployed.

[builder.py](https://github.com/hexterisk/dynamic-malwired/blob/master/builder.py): Builds the dataset from all the files in the database.

[config.py](https://github.com/hexterisk/dynamic-malwired/blob/master/config.py): Contains malware classes list that can be deleted as per requirement.

[features.py](https://github.com/hexterisk/dynamic-malwired/blob/master/features.py): Feature extraction script from the files in the dataset. Original script taken from the ember project and modified to apply various tweaks for multiclass classification and file specific signatures.

[predict.py](https://github.com/hexterisk/dynamic-malwired/blob/master/predict.py): A python script to predict a PE's class using a trained model.

[train.py](https://github.com/hexterisk/dynamic-malwired/blob/master/train.py): A python script to train the model, given the data.