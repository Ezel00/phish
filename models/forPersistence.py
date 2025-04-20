import numpy as np#
from itertools import combinations

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ripser import ripser, Rips
from persim import plot_diagrams, PersistenceImager, bottleneck
from skimage.transform import resize
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVC

'''
X_train = np.load("X_train2.npy")
X_test = np.load("X_test2.npy")
y_train = np.load("y_train2.npy")
y_test = np.load("y_test2.npy")'''

pathYTrain= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet1\y_train.npy"
pathXTest= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet1\X_test.npy"
pathXTrain= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet1\X_train.npy"
pathYTest= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet1\y_test.npy"

pathYTrain2= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet2\y_train.npy"
pathXTest2= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet2\X_test.npy"
pathXTrain2= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet2\X_train.npy"
pathYTest2= r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\featureSet2\y_test.npy"

pathXTrain3=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStatsFeatureSet1\X_train.npy"
pathYTest3=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStatsFeatureSet1\y_test.npy"
pathYTrain3=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStatsFeatureSet1\y_train.npy"
pathXTest3=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStatsFeatureSet1\X_test.npy"

pathYTest4=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStats2\y_test.npy"
pathYTrain4=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStats2\y_train.npy"
pathXTest4=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStats2\X_test.npy"
pathXTrain4=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\persistenceImages\barcodeStats2\X_train.npy"
'''
X_train = np.load(pathXTrain)
y_train=np.load(pathYTrain)
X_test = np.load(pathXTest)
y_test= np.load(pathYTest)


X_train = np.load(pathXTrain2)
y_train=np.load(pathYTrain2)
X_test = np.load(pathXTest2)
y_test= np.load(pathYTest2)

X_train = np.load(pathXTrain3)
y_train=np.load(pathYTrain3)
X_test = np.load(pathXTest3)
y_test= np.load(pathYTest3)


'''
X_train = np.load(pathXTrain4)
y_train=np.load(pathYTrain4)
X_test = np.load(pathXTest4)
y_test= np.load(pathYTest4)


def contaminate(yTra, ratio, random_state=42):
    np.random.seed(random_state)
    yNoise = yTra.copy()
    numFlip = int(len(yTra) * ratio)
    flipInd = np.random.choice(len(yTra), num_flip, replace=False)
    uniqLabe = np.unique(yTra)
    for idx in flipInd:
        current_label = yTra[idx]
        possible_labels = uniqLabe[uniqLabe != current_label]  
        yNoise[idx] = np.random.choice(possible_labels)
    return yNoise


def trainLinReg(X_train, X_test, y_train, y_test):
    scaler = RobustScaler()
    Xtrain = scaler.fit_transform(X_train)
    Xtest = scaler.transform(X_test)
    pca = PCA(n_components=.95)
    Xtrain = pca.fit_transform(Xtrain)
    Xtest = pca.transform(Xtest)
    print("PCA Transformed Shape:", Xtrain.shape)
    model = LinearRegression()
    model.fit(Xtrain, y_train)
    yP = model.predict(Xtest)
    mse = mean_squared_error(y_test, yP)
    r2 = r2_score(y_test, yP)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return model, scaler, pca

#trainLinReg(X_train, X_test, y_train, y_test)
def logReg(X_train, X_unseen, y_train, y_unseen):
    scaler = StandardScaler()
    Xtrain = scaler.fit_transform(X_train)  # Fit & transform training data
    pca = PCA(n_components=.95)
    Xtrain = pca.fit_transform(Xtrain)
    X_unseen = scaler.transform(X_unseen)  # Scale first
    X_unseen = pca.transform(X_unseen)  # Then apply PCA
    print("PCA Transformed Shape for X_train:", Xtrain.shape)
    print("PCA Transformed Shape for X_unseen:", X_unseen.shape)
    log = LogisticRegression()#(solver="saga", max_iter=5000, random_state=42)
    log.fit(Xtrain, y_train)
    yPred = log.predict(Xtrain)
    trainAcc = accuracy_score(y_train, yPred)
    print(f"Training Accuracy: {trainAcc:.4f}")
    print("Training Classification Report:\n", classification_report(y_train, yPred))
    yPred = log.predict(X_unseen)
    unseen_acc = accuracy_score(y_unseen, yPred)
    print(f" Model Evaluation on Unseen Data:")
    print(f"Accuracy: {unseen_acc:.4f}")
    print("Classification Report:\n", classification_report(y_unseen, yPred))
    return scaler, log
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
#logReg(X_train, X_test, y_train, y_test)

def knn(X_train, X_test, y_train, y_test):
    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA Transformed Shape for X_train: {X_train_pca.shape}")
    knn = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'metric': ['euclidean', 'manhattan'],
        'weights': ['uniform', 'distance']
    }
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_pca, y_train) 
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")
    best_model = KNeighborsClassifier()# grid_search.best_estimator_
    best_model.fit(X_train_pca, y_train)
    y_train_pred = best_model.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    y_test_pred = best_model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return best_model, scaler, pca


#knn(X_train, X_test, y_train, y_test)
def trainRFNoGrid(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print("\nTraining Classification Report:\n", classification_report(y_train, y_train_pred))
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

    return model, scaler


def trainRFNoGridPCA(X_train, X_test, y_train, y_test ):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA Transformed Shape for X_train: {X_train_pca.shape}")
    model  = RandomForestClassifier(random_state=42)
    #RandomForestClassifier(random_state=42, n_estimators=50,max_depth=10, min_samples_split=5, min_samples_leaf=2)
    model.fit(X_train_pca, y_train)
    y_train_pred = model.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    y_pred = model.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    return model, scaler, pca

y_train =contaminate(y_train, .3)
trainRFNoGrid(X_train, X_test, y_train, y_test)
def trainSVM(xTrain, xTest, yTrain, yTest):
        scaler = StandardScaler()
        xTrain = scaler.fit_transform(xTrain)
        pca = PCA(n_components=1)
        xTrain = pca.fit_transform(xTrain)
        print(f"PCA Transformed Shape for X_train: {xTrain.shape}")
        xTest = scaler.transform(xTest)
        xTest = pca.transform(xTest)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        #grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5)
        #grid_search.fit(xTrain, yTrain)
        #print(f"Best parameters: {grid_search.best_params_}")
        model = SVC()#grid_search.best_estimator_
        model.fit(xTrain, yTrain)
        y_train_pred = model.predict(xTrain)
        train_accuracy = accuracy_score(yTrain, y_train_pred)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        yPred = model.predict(xTest)
        accuracy = accuracy_score(yTest, yPred)
        print(f"Accuracy for the test data: {accuracy:.4f}")
        #print("Classification Report:\n", classification_report(yTest, yPred))
        return model, scaler#, pca

#trainSVM(X_train, X_test, y_train, y_test)
import xgboost as xgb


def trainXG(xTrain, xTest, yTrain, yTest):
    model= xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model=xgb.XGBClassifier(learning_rate=0.05, n_estimators=1000)
    pca = PCA(n_components=150)
    '''
    xTrain = pca.fit_transform(xTrain)
    print(f"PCA Transformed Shape for X_train: {xTrain.shape}")
    xTest = pca.transform(xTest)'''
    model.fit(xTrain, yTrain)
    yTrainPred = model.predict(xTrain)
    train_acc = accuracy_score(yTrain, yTrainPred)
    yTestPred = model.predict(xTest)
    test_acc = accuracy_score(yTest, yTestPred)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}\n")
    print("Test Classification Report:\n", classification_report(yTest, yTestPred))
    return model

#trainXG(X_train, X_test, y_train, y_test)

#y_train = contaminate(y_train, .3)
#trainXG(X_train, X_test, y_train, y_test)


