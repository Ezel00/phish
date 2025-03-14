import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

pathEmbedTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\testEMBED.csv"
pathEmbedTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\trainEMBED.csv"
pathTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA8Test.csv"
pathTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet2\dfPCA8Train.csv"
'''
train = pd.read_csv(pathEmbedTrain)  #pd.read_csv(pathTrain)
test =  pd.read_csv(pathEmbedTest) #pd.read_csv(pathTest)
'''
train = pd.read_csv(pathTrain)
test =  pd.read_csv(pathTest)
XTrain = train.drop(columns="label")
yTrain = train["label"]
XTest =test.drop(columns="label")
yTest = test["label"]

def train(X, y, n):
    scaler = StandardScaler()
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    XTrain = scaler.fit_transform(XTrain)
    model = KNeighborsClassifier(n)
    model.fit(XTrain, yTrain)
    yPred = model.predict(XTest)
    cv_scores = cross_val_score(model, XTrain, yTrain, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy (mean): {np.mean(cv_scores):.4f}")
    accuracy = accuracy_score(yTest, yPred)
    report = classification_report(yTest, yPred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    return model, scaler

def test(model, scaler, XTest, yTest):
    XTest = scaler.transform(XTest)
    yPred = model.predict(XTest)
    accuracy = accuracy_score(yTest, yPred)
    report = classification_report(yTest, yPred)
    print(f"Model Accuracy on Unseen Data: {accuracy:.4f} \n")
    print("Classification Report:")
    print(report)

'''
# Trying different n_neighbors values
for n_neighbors in [1, 3, 5, 7, 9]:
    print(f"Training with n_neighbors = {n_neighbors}")
    knnModel, scaler = trainKnnModelWithCrossValidation(XTrain, yTrain, n_neighbors)
    testKnnModel(knnModel, scaler, XTest, yTest)
'''

knnModel, scaler = train(XTrain, yTrain, 9)
test(knnModel, scaler, XTest, yTest)
