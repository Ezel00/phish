from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


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
    scaler = StandardScaler()
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    xTrain = scaler.fit_transform(xTrain)
    model = SVC(kernel='linear', random_state=42)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy for the training data: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(yTest, yPred))
    return model, scaler

def test(model, scaler, xTest, yTest):
    xTest = scaler.transform(xTest)
    yPred = model.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy for the unseens data: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(yTest, yPred))

model, scaler = train(XTrain, yTrain)
test(model, scaler, XTest, yTest)
