import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

pathEmbedTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\test.csv"
pathEmbedTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\train.csv"
pathTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Test.csv"
pathTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Train.csv"
'''
train = pd.read_csv(pathEmbedTrain)  #pd.read_csv(pathTrain)
test =  pd.read_csv(pathEmbedTest) #pd.read_csv(pathTest)
#accuracy for embed big ones Accuracy: 0.9175

'''
train = pd.read_csv(pathTrain)
test =  pd.read_csv(pathTest)
XTrain = train.drop(columns="label")
yTrain = train["label"]
XTest =test.drop(columns="label")
yTest = test["label"]

def dirtyY(yT, flip, random_seed=42):
    np.random.seed(random_seed)
    yT = np.array(yT)
    zeros = np.where(yT == 0)[0]
    ones = np.where(yT == 1)[0]
    num_zeros = int(len(zeros) * flip)
    num_ones = int(len(ones) * flip)
    zero_inds = np.random.choice(zeros, num_zeros, replace=False)
    one_inds = np.random.choice(ones, num_ones, replace=False)
    yT[zero_inds] = 1  # Flip 0 -> 1
    #yT[one_inds] = 0  # Flip 1 -> 0
    print(f"Original Label Distribution: 0s = {len(zeros)}, 1s = {len(ones)}")
    new_zeros = np.sum(yT == 0)
    new_ones = np.sum(yT == 1)
    print(f"Dirty Label Distribution: 0s = {new_zeros}, 1s = {new_ones}")
    return yT




def logReg(X,y):
    X_train, X_unseen, y_train, y_unseen = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(X_train)
    log = LogisticRegression(random_state=42) #LogisticRegression(random_state=42, max_iter=2000, solver='saga',tol=1e-4)  #
    log.fit(Xtrain, y_train)
    yPred = log.predict(Xtrain)
    trainAcc = accuracy_score(y_train, yPred)
    print(f"Training Accuracy: {trainAcc:.4f}")
    print("Training Classification Report:\n", classification_report(y_train, yPred))
    return scaler, log

def test(X,y, scaler, log):
    X = scaler.transform(X)
    yPred= log.predict(X)
    unseen_acc = accuracy_score(y, yPred)
    print(f" Model Evaluation on Unseen Data:")
    print(f"Accuracy: {unseen_acc:.4f}")
    print("Classification Report:\n", classification_report(y, yPred))


def crossVal(X, y , cv):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver='lbfgs', random_state=42))
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Cross-Validation Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores

yTrain = dirtyY(yTrain, 0.4, 42)
crossVal(XTrain , yTrain , 5)
scaler, log=logReg(XTrain,yTrain)
test(XTest,yTest,scaler,log)
