#svm for use on barcode statistics
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

p=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\trainBarcodeStats.csv"
train = pd.read_csv(p)

p2=r"C:\Users\ezele\Desktop\thesis\tdaPython\final\landmarks\statDatasets\testBarcodeStats.csv"
test = pd.read_csv(p2)

XTrainEmbed = train.drop(columns="label")
yTrainEmbed = train["label"]
xTest=test.drop(columns="label")
yTest=test["label"]

def train(X, y):
        scaler = StandardScaler()
        xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
        xTrain = scaler.fit_transform(xTrain)
        pca = PCA(n_components=0.95)
        xTrain = pca.fit_transform(xTrain)
        xTest = scaler.transform(xTest)
        xTest = pca.transform(xTest)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=5)
        grid_search.fit(xTrain, yTrain)
        print(f"Best parameters: {grid_search.best_params_}")
        model = grid_search.best_estimator_
        model.fit(xTrain, yTrain)
        yPred = model.predict(xTest)
        accuracy = accuracy_score(yTest, yPred)
        print(f"Accuracy for the test data: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(yTest, yPred))
        return model, scaler, pca


def test(model, scaler, pca, xTest, yTest):
    xTest = scaler.transform(xTest)
    xTest = pca.transform(xTest)  # Apply PCA to new data
    yPred = model.predict(xTest)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy for the unseen data: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(yTest, yPred))


model, scaler, pca = train(XTrainEmbed, yTrainEmbed)
test(model, scaler, pca, xTest, yTest)
