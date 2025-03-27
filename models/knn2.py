#THIS IS THE KNN FOR BARCODE STATISTICS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

p1= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\testBarcodeStats.csv"
p2=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\trainBarcodeStats.csv"
pTest = r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\testEmbedBarcodeStats.csv"
pTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\trainEmbedBarcodeStats.csv"


train = pd.read_csv(pTrain)
test =  pd.read_csv(pTest)
XTrain = train.drop(columns="label")
yTrain = train["label"]
XTest =test.drop(columns="label")
yTest = test["label"]


def train(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    print(X_pca.shape)
    XTrain, X_test, yTrain, yTest = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance']
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(XTrain, yTrain)
    bestP = grid_search.best_params_
    print(f"Best Parameters: {bestP}")
    model = grid_search.best_estimator_
    yPred = model.predict(XTest)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.4f}")
    #print(classification_report(y_test, y_pred))
    return model, scaler, pca


def test(model, scaler, pca, XTest, yTest):
    xScaled = scaler.transform(XTest) 
    xScaled = pca.transform(xScaled)  
    yPred = model.predict(xScaled)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy:.4f}")
    #print(classification_report(yTest, yPred))

'''
model, scaler, pca =t2(XTrain, yTrain)
test(model, scaler, pca, XTest, yTest)'''


yT= yTrain#dirtyY(yTrain, .4)
model, scaler, pca =train(XTrain, yT)
test(model, scaler, pca, XTest, yTest)
