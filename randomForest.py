
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

pathEmbedTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\testEMBED.csv"
pathEmbedTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\trainEMBED.csv"
pathTest=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Test.csv"
pathTrain=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\pcaDATASETS\FeatureSet1\dfPCA12Train.csv"
def trainRF(X_train, yTrain):
    scaler = StandardScaler()
    XTrain = scaler.fit_transform(X_train)
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees
        'max_depth': [None, 10, 20],  # Tree depth
        'min_samples_split': [2, 5, 10],  # Minimum samples per split
        'min_samples_leaf': [1, 2, 4],  # Minimum samples per leaf
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(XTrain, yTrain)
    model = grid_search.best_estimator_
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    return model, scaler

def trainRFNoGrid(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,
                                                        random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Model Accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
    return model, scaler
#for not embeddings
#Best Hyperparameters: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
#pca12
#Best Hyperparameters: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}
def testUnseen(model, scaler, XTest, yTest):
    XTest_scaled = scaler.transform(XTest)
    yPred = model.predict(XTest_scaled)
    print("\nModel Evaluation on Unseen Data:")
    print("Accuracy:", accuracy_score(yTest, yPred))
    print("Classification Report:\n", classification_report(yTest, yPred))


train = pd.read_csv(pathTrain)
test = pd.read_csv(pathTest)
XTrain = train.drop(columns="label")
yTrain = train["label"]
XTest =test.drop(columns="label")
yTest = test["label"]
# Train the Random Forest model
print(train.shape, test.shape)
modelT, scalerT = trainRFNoGrid(XTrain, yTrain)

# Evaluate on unseen test data
testUnseen(modelT, scalerT, XTest, yTest)
