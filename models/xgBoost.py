
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


pathEmbedTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\test.csv"
pathEmbedTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\train.csv"
pathTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Test.csv"
pathTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Train.csv"
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

def train(X, y):
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    acc = accuracy_score(yTest, yPred)
    print(f"Accuracy for training data: {acc:.4f}\n")
    print(f"Classification Report:\n{classification_report(yTest, yPred)}")

    return model
  
def test(model, xTest, yTest):
    yPred = model.predict(xTest)
    acc = accuracy_score(yTest, yPred)
    print(f"Accuracy for unseen data: {acc:.4f}\n")
    print(f"Classification Report:\n{classification_report(yTest, yPred)}")

xgbModel = train(XTrain, yTrain)
test(xgbModel, XTest, yTest)
